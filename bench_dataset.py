"""Donut dataset benchmark.

Loads a pool of images from a HuggingFace dataset (or local directory), then
runs N batched inference passes sampling randomly from the pool. Reports latency
distributions (mean ± std, p50/p95/p99) for preprocess, encode, decode, and
end-to-end, plus throughput in samples/sec and tokens/sec.

All latency metrics are per-batch. With --batch-size 4 and --runs 50 you get 50
timing measurements each covering 4 images. Decode time is dominated by the
longest sequence in the batch (padding effect).

Usage:
    uv run bench_dataset.py [--pool 20] [--runs 50] [--batch-size 1]
    uv run bench_dataset.py --dataset naver-clova-ix/cord-v2 --pool 50 --runs 100
    uv run bench_dataset.py --image-dir /path/to/images --pool 20 --runs 50
    uv run bench_dataset.py --batch-size 4 --save results.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from patches import patch_attn_mask, patch_attn_mask_gpu

MODEL_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"
TASK_PROMPT = "<s_cord-v2>"
DEFAULT_DATASET = "naver-clova-ix/cord-v2"


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class CudaTimer:
    """Wall-clock timer with cuda sync. Simple and correct for both CPU and GPU."""

    def __init__(self):
        self._start = None

    def start(self):
        cuda_sync()
        self._start = time.perf_counter()

    def stop(self) -> float:
        cuda_sync()
        return (time.perf_counter() - self._start) * 1000.0  # ms


# ── Data loading ─────────────────────────────────────────────────────────────


def load_pool(args) -> list[Image.Image]:
    if args.image_dir:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        paths = sorted(
            p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in extensions
        )
        if not paths:
            sys.exit(f"No images found in {args.image_dir}")
        images = [Image.open(p).convert("RGB") for p in paths[: args.pool]]
        if len(paths) < args.pool:
            print(
                f"  pool        : {len(images)} images (dir has {len(paths)}; capped from requested {args.pool})"
            )
        else:
            print(f"  pool        : {len(images)} images (from {args.image_dir})")
    else:
        from datasets import (
            load_dataset,
        )  # lazy import — not needed for --image-dir path

        ds = load_dataset(args.dataset, split=args.dataset_split)
        available = len(ds)
        n = min(args.pool, available)
        images = [ds[i][args.image_column].convert("RGB") for i in range(n)]
        if available < args.pool:
            print(
                f"  pool        : {n} images (dataset has {available}; capped from requested {args.pool})"
            )
        else:
            print(
                f"  pool        : {n} images (from {args.dataset} / {args.dataset_split})"
            )
    if not images:
        sys.exit("Pool is empty — check --dataset / --image-dir / --pool")
    return images


def sample_batch(pool: list[Image.Image], batch_size: int) -> list[Image.Image]:
    return [pool[random.randrange(len(pool))] for _ in range(batch_size)]


# ── Inference ────────────────────────────────────────────────────────────────


def run_once_batched(
    model,
    processor,
    images: list[Image.Image],
    dev: str,
) -> tuple[float, float, float, float, int]:
    """One batched encode+decode pass.

    Returns (preprocess_ms, encode_ms, decode_ms, e2e_ms, n_tokens_sum).
    All latency values are per-batch (not per-image).
    n_tokens_sum is the total tokens generated across all images in the batch.
    """
    t_e2e = CudaTimer()
    t_e2e.start()

    # Preprocess (CPU — bare wall-clock, no cuda_sync needed)
    t0 = time.perf_counter()
    pixel_values = (
        processor(images, return_tensors="pt").pixel_values.to(dev).to(model.dtype)
    )
    decoder_input_ids = processor.tokenizer(
        TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(dev)
    # Expand decoder_input_ids for each item in the batch
    decoder_input_ids = decoder_input_ids.expand(len(images), -1)
    preprocess_ms = (time.perf_counter() - t0) * 1000.0

    t = CudaTimer()

    # Encode
    t.start()
    with torch.no_grad():
        enc_out = model.encoder(pixel_values, return_dict=True)
    encode_ms = t.stop()

    # Decode
    t.start()
    with torch.no_grad():
        seqs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=enc_out,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        ).sequences
    decode_ms = t.stop()

    e2e_ms = t_e2e.stop()

    # Count tokens generated (sum across batch, each up to EOS)
    prompt_len = decoder_input_ids.shape[1]
    n_tokens_sum = 0
    for row in seqs[:, prompt_len:]:
        eos = (row == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        n_tokens_sum += int(eos[0].item()) + 1 if len(eos) > 0 else len(row)

    return preprocess_ms, encode_ms, decode_ms, e2e_ms, n_tokens_sum


# ── Statistics ───────────────────────────────────────────────────────────────


def percentiles(values: list[float]) -> tuple[float, float, float]:
    if len(values) < 2:
        v = values[0] if values else 0.0
        return v, v, v
    q = statistics.quantiles(values, n=100, method="inclusive")
    return q[49], q[94], q[98]


def fmt(values: list[float]) -> str:
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{m:7.1f} ± {s:5.1f} ms"


def _system_info() -> dict:
    info: dict = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = str(torch.backends.cudnn.version())
    return info


def save_stats(
    path: str,
    args,
    pool: list[Image.Image],
    input_shape: list[int],
    preprocess_times: list[float],
    enc_times: list[float],
    dec_times: list[float],
    e2e_times: list[float],
    token_counts: list[int],
    peak_mb: float,
) -> None:
    def _stat(vals: list[float]) -> dict:
        p50, p95, p99 = percentiles(vals)
        return {
            "mean": round(statistics.mean(vals), 3),
            "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 3),
            "p50": round(p50, 3),
            "p95": round(p95, 3),
            "p99": round(p99, 3),
        }

    mean_e2e_s = statistics.mean(e2e_times) / 1000.0
    mean_dec_s = statistics.mean(dec_times) / 1000.0
    mean_tok = statistics.mean(token_counts)

    data = {
        "schema_version": 1,
        "script": "bench_dataset.py",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "args": {
            "model": args.model,
            "device": args.device,
            "warmup": args.warmup,
            "runs": args.runs,
            "pool": args.pool,
            "batch_size": args.batch_size,
            "dataset": getattr(args, "dataset", None),
            "dataset_split": getattr(args, "dataset_split", None),
            "image_column": getattr(args, "image_column", None),
            "image_dir": args.image_dir,
            "no_patch": args.no_patch,
        },
        "system": _system_info(),
        "pool": {
            "size": len(pool),
            "image_sizes": [[img.width, img.height] for img in pool],
        },
        "input_shape": input_shape,
        "runs": {
            "preprocess_ms": preprocess_times,
            "encode_ms": enc_times,
            "decode_ms": dec_times,
            "e2e_ms": e2e_times,
            "n_tokens": token_counts,
        },
        "summary": {
            "preprocess_ms": _stat(preprocess_times),
            "encode_ms": _stat(enc_times),
            "decode_ms": _stat(dec_times),
            "e2e_ms": _stat(e2e_times),
            "samples_per_sec": round(args.batch_size / mean_e2e_s, 3),
            "tokens_per_sec": round(mean_tok / mean_dec_s, 2),
            "tokens_mean": round(mean_tok, 1),
            "peak_gpu_mb": round(peak_mb, 1),
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  stats saved → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Donut dataset benchmark")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument(
        "--pool", type=int, default=20, help="number of images to load into pool"
    )
    parser.add_argument("--batch-size", type=int, default=1, dest="batch_size")

    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--dataset", default=DEFAULT_DATASET, help="HuggingFace dataset name"
    )
    src.add_argument(
        "--image-dir", default=None, dest="image_dir", help="local directory of images"
    )

    parser.add_argument("--dataset-split", default="test", dest="dataset_split")
    parser.add_argument("--image-column", default="image", dest="image_column")
    parser.add_argument("--no-patch", action="store_true")
    parser.add_argument(
        "--save", default=None, metavar="PATH", help="save per-run stats to JSON"
    )
    args = parser.parse_args()

    # When --image-dir is used the dataset/split/column args are irrelevant but
    # argparse still sets them; that is fine.

    dev = args.device

    # ── System info ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Donut dataset benchmark")
    print("=" * 60)
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  Transformers: {transformers.__version__}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA        : {torch.version.cuda}")
    print(f"  Device      : {dev}")

    # ── Load model ───────────────────────────────────────────────────────────
    print("\nLoading model...")
    processor = DonutProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    )
    model.to(dev).eval()

    enc_dtype = next(model.encoder.parameters()).dtype
    print(f"  dtype       : {enc_dtype}")

    if args.no_patch:
        print("  patch       : DISABLED (--no-patch)")
    elif dev == "cuda":
        patch_attn_mask_gpu(model)
        print("  patch       : applied (gpu direct)")
    else:
        patch_attn_mask(model)
        print("  patch       : applied (cpu float32)")

    # ── Load pool ────────────────────────────────────────────────────────────
    print("\nLoading image pool...")
    pool = load_pool(args)

    # ── Dry-run to determine input shape ─────────────────────────────────────
    sample_imgs = sample_batch(pool, args.batch_size)
    sample_pv = (
        processor(sample_imgs, return_tensors="pt").pixel_values.to(dev).to(model.dtype)
    )
    input_shape = list(sample_pv.shape)
    print(f"  batch shape : {input_shape}")
    if args.batch_size > 1:
        print(
            "  note        : decode time dominated by longest sequence in batch (padding)"
        )
    del sample_pv

    # ── Warmup ───────────────────────────────────────────────────────────────
    print(f"\nWarmup ({args.warmup} run(s))...")
    for _ in range(args.warmup):
        imgs = sample_batch(pool, args.batch_size)
        run_once_batched(model, processor, imgs, dev)

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ── Measurement loop ─────────────────────────────────────────────────────
    preprocess_times: list[float] = []
    enc_times: list[float] = []
    dec_times: list[float] = []
    e2e_times: list[float] = []
    token_counts: list[int] = []

    for _ in range(args.runs):
        imgs = sample_batch(pool, args.batch_size)
        pre_ms, enc_ms, dec_ms, e2e_ms, n_tok = run_once_batched(
            model, processor, imgs, dev
        )
        preprocess_times.append(pre_ms)
        enc_times.append(enc_ms)
        dec_times.append(dec_ms)
        e2e_times.append(e2e_ms)
        token_counts.append(n_tok)

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if dev == "cuda" else 0.0

    mean_e2e_s = statistics.mean(e2e_times) / 1000.0
    mean_dec_s = statistics.mean(dec_times) / 1000.0
    mean_tok = statistics.mean(token_counts)
    samples_per_sec = args.batch_size / mean_e2e_s
    tok_per_sec = mean_tok / mean_dec_s

    # ── Results ──────────────────────────────────────────────────────────────
    w = 18
    print(f"\nN={args.runs} runs, batch={args.batch_size}")
    print(f"  {'preprocess':<{w}} : {fmt(preprocess_times)}")
    print(f"  {'encode':<{w}} : {fmt(enc_times)}")
    print(f"  {'decode':<{w}} : {fmt(dec_times)}")
    print(f"  {'end-to-end':<{w}} : {fmt(e2e_times)}")

    print("\nPercentiles (ms):")
    pw = 12
    for label, vals in [
        ("preprocess", preprocess_times),
        ("encode", enc_times),
        ("decode", dec_times),
        ("end-to-end", e2e_times),
    ]:
        p50, p95, p99 = percentiles(vals)
        print(f"  {label:<{pw}}  p50={p50:7.1f}  p95={p95:7.1f}  p99={p99:7.1f}")

    print("\nThroughput:")
    print(f"  {'samples/sec':<{w}} : {samples_per_sec:7.2f}")
    print(f"  {'tokens/sec':<{w}} : {tok_per_sec:7.1f}")
    print(f"  {'tokens (mean/run)':<{w}} : {mean_tok:7.0f}")
    if dev == "cuda":
        print(f"  {'peak GPU mem':<{w}} : {peak_mb:7.0f} MB")

    if args.save:
        save_stats(
            args.save,
            args,
            pool,
            input_shape,
            preprocess_times,
            enc_times,
            dec_times,
            e2e_times,
            token_counts,
            peak_mb,
        )


if __name__ == "__main__":
    main()
