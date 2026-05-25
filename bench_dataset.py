"""Donut dataset benchmark — high-level component timing.

Measures preprocessing, encoder, and decoder latency over a pool of real images
from a HuggingFace dataset (or local directory).

Key metric: decode_ms_per_token (total decode time / tokens generated).
Raw decode_ms is not representative because the decoder is autoregressive and
its wall-clock time grows linearly with sequence length.

Prints a 5-line summary; saves full per-run raw data to JSON with --save.

Usage:
    uv run bench_dataset.py [--pool 50] [--runs 50] [--batch-size 1]
    uv run bench_dataset.py --dataset naver-clova-ix/cord-v2 --pool 50
    uv run bench_dataset.py --image-dir /path/to/images --pool 20 --runs 50
    uv run bench_dataset.py --batch-size 4 --max-new-tokens 100 --save out.json
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
    """Wall-clock timer with cuda sync. Correct for both CPU and GPU."""

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
    else:
        from datasets import load_dataset

        ds = load_dataset(args.dataset, split=args.dataset_split)
        n = min(args.pool, len(ds))
        images = [ds[i][args.image_column].convert("RGB") for i in range(n)]
    if not images:
        sys.exit("Pool is empty — check --dataset / --image-dir / --pool")
    return images


def sample_batch(pool: list[Image.Image], batch_size: int) -> list[Image.Image]:
    return [pool[random.randrange(len(pool))] for _ in range(batch_size)]


# ── Inference ────────────────────────────────────────────────────────────────


def run_once(
    model,
    processor,
    images: list[Image.Image],
    dev: str,
    max_new_tokens: int | None,
) -> tuple[float, float, float, float, int, float, float]:
    """One encode+decode pass.

    Returns (preprocess_ms, encode_ms, decode_ms, e2e_ms, n_tokens,
             encode_peak_mb, decode_peak_mb).
    All latency values are per-batch. n_tokens is total tokens generated
    across all images in the batch.
    """
    t_e2e = CudaTimer()
    t_e2e.start()

    # Preprocess (CPU)
    t0 = time.perf_counter()
    pixel_values = (
        processor(images, return_tensors="pt").pixel_values.to(dev).to(model.dtype)
    )
    decoder_input_ids = processor.tokenizer(
        TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(dev)
    decoder_input_ids = decoder_input_ids.expand(len(images), -1)
    preprocess_ms = (time.perf_counter() - t0) * 1000.0

    t = CudaTimer()

    # Encode — track peak memory for this phase
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t.start()
    with torch.no_grad():
        enc_out = model.encoder(pixel_values, return_dict=True)
    encode_ms = t.stop()
    encode_peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if dev == "cuda" else 0.0

    # Decode — reset and track peak memory separately
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gen_kwargs: dict = dict(
        pixel_values=pixel_values,
        decoder_input_ids=decoder_input_ids,
        encoder_outputs=enc_out,
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    if max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = max_new_tokens
        del gen_kwargs["max_length"]
    t.start()
    with torch.no_grad():
        seqs = model.generate(**gen_kwargs).sequences
    decode_ms = t.stop()
    decode_peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if dev == "cuda" else 0.0

    e2e_ms = t_e2e.stop()

    # Token count (sum across batch, each up to EOS)
    prompt_len = decoder_input_ids.shape[1]
    n_tokens = 0
    for row in seqs[:, prompt_len:]:
        eos = (row == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        n_tokens += int(eos[0].item()) + 1 if len(eos) > 0 else len(row)

    return preprocess_ms, encode_ms, decode_ms, e2e_ms, n_tokens, encode_peak_mb, decode_peak_mb


# ── Statistics ───────────────────────────────────────────────────────────────


def _stat(vals: list[float]) -> dict:
    if len(vals) < 2:
        v = vals[0] if vals else 0.0
        return {"mean": round(v, 3), "std": 0.0, "p50": round(v, 3), "p95": round(v, 3), "p99": round(v, 3)}
    q = statistics.quantiles(vals, n=100, method="inclusive")
    return {
        "mean": round(statistics.mean(vals), 3),
        "std": round(statistics.stdev(vals), 3),
        "p50": round(q[49], 3),
        "p95": round(q[94], 3),
        "p99": round(q[98], 3),
    }


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


def save_results(path: str, args, pool: list[Image.Image], input_shape: list[int], runs: dict) -> None:
    pre = runs["preprocess_ms"]
    enc = runs["encode_ms"]
    dec = runs["decode_ms"]
    e2e = runs["e2e_ms"]
    tok = runs["n_tokens"]
    dec_per_tok = runs["decode_ms_per_token"]

    mean_e2e_s = statistics.mean(e2e) / 1000.0
    mean_tok = statistics.mean(tok)

    data = {
        "schema_version": 2,
        "script": "bench_dataset.py",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "args": {
            "model": args.model,
            "device": args.device,
            "warmup": args.warmup,
            "runs": args.runs,
            "pool": args.pool,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "dataset": getattr(args, "dataset", None),
            "dataset_split": getattr(args, "dataset_split", None),
            "image_dir": args.image_dir,
            "no_patch": args.no_patch,
        },
        "system": _system_info(),
        "pool": {
            "size": len(pool),
            "image_sizes": [[img.width, img.height] for img in pool],
        },
        "input_shape": input_shape,
        "runs": runs,
        "summary": {
            "preprocess_ms": _stat(pre),
            "encode_ms": _stat(enc),
            "decode_ms": _stat(dec),
            "decode_ms_per_token": _stat(dec_per_tok),
            "e2e_ms": _stat(e2e),
            "n_tokens": _stat([float(t) for t in tok]),
            "encode_peak_gpu_mb": round(max(runs["encode_peak_mb"]), 1),
            "decode_peak_gpu_mb": round(max(runs["decode_peak_mb"]), 1),
            "samples_per_sec": round(args.batch_size / mean_e2e_s, 3),
            "tokens_per_sec": round(mean_tok / (statistics.mean(dec) / 1000.0), 2),
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  saved → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Donut dataset benchmark (high-level)")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--pool", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1, dest="batch_size")
    parser.add_argument("--max-new-tokens", type=int, default=None, dest="max_new_tokens",
                        help="cap decoder generation length (default: uncapped)")

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--dataset", default=DEFAULT_DATASET)
    src.add_argument("--image-dir", default=None, dest="image_dir")

    parser.add_argument("--dataset-split", default="test", dest="dataset_split")
    parser.add_argument("--image-column", default="image", dest="image_column")
    parser.add_argument("--no-patch", action="store_true")
    parser.add_argument("--save", default=None, metavar="PATH")
    args = parser.parse_args()

    dev = args.device

    print(f"Loading model ({args.model})...")
    processor = DonutProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.to(dev).eval()

    if args.no_patch:
        pass
    elif dev == "cuda":
        patch_attn_mask_gpu(model)
    else:
        patch_attn_mask(model)

    print("Loading image pool...")
    pool = load_pool(args)

    sample_imgs = sample_batch(pool, args.batch_size)
    sample_pv = processor(sample_imgs, return_tensors="pt").pixel_values.to(dev).to(model.dtype)
    input_shape = list(sample_pv.shape)
    del sample_pv

    print(f"Warmup ({args.warmup} run(s))...")
    for _ in range(args.warmup):
        run_once(model, processor, sample_batch(pool, args.batch_size), dev, args.max_new_tokens)

    print(f"Measuring ({args.runs} run(s), batch={args.batch_size})...")
    pre_list: list[float] = []
    enc_list: list[float] = []
    dec_list: list[float] = []
    e2e_list: list[float] = []
    tok_list: list[int] = []
    dec_per_tok_list: list[float] = []
    enc_peak_list: list[float] = []
    dec_peak_list: list[float] = []

    for _ in range(args.runs):
        imgs = sample_batch(pool, args.batch_size)
        pre, enc, dec, e2e, n_tok, enc_peak, dec_peak = run_once(
            model, processor, imgs, dev, args.max_new_tokens
        )
        pre_list.append(pre)
        enc_list.append(enc)
        dec_list.append(dec)
        e2e_list.append(e2e)
        tok_list.append(n_tok)
        dec_per_tok_list.append(dec / n_tok if n_tok > 0 else 0.0)
        enc_peak_list.append(enc_peak)
        dec_peak_list.append(dec_peak)

    # ── Summary print ────────────────────────────────────────────────────────
    def _mean_std(vals: list[float], unit: str = "ms") -> str:
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{m:7.1f} ± {s:5.1f} {unit}"

    w = 12
    tok_mean = statistics.mean(tok_list)
    tok_std = statistics.stdev(tok_list) if len(tok_list) > 1 else 0.0
    tok_per_sec = tok_mean / (statistics.mean(dec_list) / 1000.0)

    print()
    print("=" * 55)
    print(f"  {'device':<{w}} : {dev}  |  batch={args.batch_size}  |  N={args.runs}")
    if torch.cuda.is_available():
        print(f"  {'GPU':<{w}} : {torch.cuda.get_device_name(0)}")
    print("=" * 55)
    enc_peak_str = f"  [peak {statistics.mean(enc_peak_list):.0f} MB]" if dev == "cuda" else ""
    dec_peak_str = f"  [peak {statistics.mean(dec_peak_list):.0f} MB]" if dev == "cuda" else ""
    print(f"  {'preprocess':<{w}} : {_mean_std(pre_list)}")
    print(f"  {'encode':<{w}} : {_mean_std(enc_list)}{enc_peak_str}")
    print(f"  {'decode':<{w}} : {_mean_std(dec_per_tok_list, 'ms/tok')}{dec_peak_str}")
    print(f"  {'tokens':<{w}} : {tok_mean:.0f} ± {tok_std:.0f}  (range {min(tok_list)}–{max(tok_list)})")
    print(f"  {'throughput':<{w}} : {tok_per_sec:.0f} tok/s  |  {args.batch_size / (statistics.mean(e2e_list) / 1000.0):.2f} samples/s")
    print("=" * 55)

    if args.save:
        save_results(
            args.save,
            args,
            pool,
            input_shape,
            {
                "preprocess_ms": pre_list,
                "encode_ms": enc_list,
                "decode_ms": dec_list,
                "decode_ms_per_token": dec_per_tok_list,
                "e2e_ms": e2e_list,
                "n_tokens": tok_list,
                "encode_peak_mb": enc_peak_list,
                "decode_peak_mb": dec_peak_list,
            },
        )


if __name__ == "__main__":
    main()
