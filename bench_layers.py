"""Donut layer-level profiler — low-level component timing over a dataset.

Times each encoder stage (Swin) and each decoder layer (MBart) using CUDA events,
aggregated across N images from a real dataset.

Encoder metrics: absolute ms per stage (independent of sequence length).
Decoder metrics: ms per token per layer (normalized, since the decoder is
autoregressive and each hook fires once per generated token).

Prints a compact two-table summary; saves full per-image raw data to JSON.

Requires CUDA. Use bench.py / diagnose.py for CPU profiling.

Usage:
    uv run bench_layers.py [--n-images 10] [--pool 20] [--save layers.json]
    uv run bench_layers.py --dataset naver-clova-ix/cord-v2 --n-images 20
    uv run bench_layers.py --image-dir /path/to/images --n-images 5
    uv run bench_layers.py --no-patch --save layers.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import random
import statistics
import sys
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from patches import patch_attn_mask_gpu

MODEL_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"
TASK_PROMPT = "<s_cord-v2>"
DEFAULT_DATASET = "naver-clova-ix/cord-v2"


def cuda_sync():
    torch.cuda.synchronize()


# ── Layer timer via CUDA events ───────────────────────────────────────────────


class LayerTimer:
    """Collects per-call GPU timing via CUDA event pairs attached as hooks."""

    def __init__(self, name: str):
        self.name = name
        self.events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._start: torch.cuda.Event | None = None

    def pre_hook(self, module, input):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._start = e

    def post_hook(self, module, input, output):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.events.append((self._start, e))

    def reset(self) -> None:
        self.events.clear()
        self._start = None

    def elapsed_no_sync(self) -> list[float]:
        """Read elapsed times. Caller must have called torch.cuda.synchronize() first."""
        return [s.elapsed_time(end) for s, end in self.events]


def register_timers(model) -> tuple[list[LayerTimer], list]:
    """Attach LayerTimers to encoder stages and decoder layers."""
    timers: list[LayerTimer] = []
    handles: list = []

    def attach(module, name: str) -> LayerTimer:
        t = LayerTimer(name)
        handles.append(module.register_forward_pre_hook(t.pre_hook))
        handles.append(module.register_forward_hook(t.post_hook))
        timers.append(t)
        return t

    attach(model.encoder.embeddings, "enc:patch_embed")
    for i, stage in enumerate(model.encoder.encoder.layers):
        attach(stage, f"enc:stage{i}")

    attach(model.decoder.model.decoder.embed_tokens, "dec:embed_tokens")
    for i, layer in enumerate(model.decoder.model.decoder.layers):
        attach(layer, f"dec:layer{i}")
    attach(model.decoder.lm_head, "dec:lm_head")

    return timers, handles


def remove_handles(handles: list) -> None:
    for h in handles:
        h.remove()


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


# ── Inference ────────────────────────────────────────────────────────────────


def run_pass(model, processor, pixel_values, decoder_input_ids):
    with torch.no_grad():
        enc_out = model.encoder(pixel_values, return_dict=True)
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
    return enc_out, seqs


def profile_one_image(
    model,
    processor,
    image: Image.Image,
    dev: str,
    timers: list[LayerTimer],
) -> dict:
    """Run one profiled pass and return per-layer timing for this image."""
    for t in timers:
        t.reset()

    pixel_values = (
        processor(image, return_tensors="pt").pixel_values.to(dev).to(model.dtype)
    )
    decoder_input_ids = processor.tokenizer(
        TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(dev)

    _, seqs = run_pass(model, processor, pixel_values, decoder_input_ids)

    # Single sync, then read all CUDA event times
    torch.cuda.synchronize()
    by_name: dict[str, list[float]] = {t.name: t.elapsed_no_sync() for t in timers}

    # Token count
    prompt_len = decoder_input_ids.shape[1]
    row = seqs[0, prompt_len:]
    eos = (row == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    n_tokens = int(eos[0].item()) + 1 if len(eos) > 0 else len(row)

    # Encoder: one call each, absolute ms
    enc_keys = [k for k in by_name if k.startswith("enc:")]
    encoder: dict = {}
    for key in enc_keys:
        field = key.removeprefix("enc:")
        encoder[field] = round(by_name[key][0], 3)
    encoder["total"] = round(sum(encoder[v] for v in encoder), 3)

    # Decoder: per-token ms per layer
    dec_keys = [k for k in by_name if k.startswith("dec:")]
    decoder: dict = {"layers": {}}
    for key in dec_keys:
        samples = by_name[key]
        total_ms = sum(samples)
        label = key.removeprefix("dec:")
        decoder["layers"][label] = {
            "ms_per_token": round(total_ms / n_tokens if n_tokens > 0 else 0.0, 4),
            "total_ms": round(total_ms, 3),
            "samples": [round(v, 4) for v in samples],
        }
    dec_total_ms = sum(v["total_ms"] for v in decoder["layers"].values())
    decoder["total_ms_per_token"] = round(dec_total_ms / n_tokens if n_tokens > 0 else 0.0, 4)

    return {"n_tokens": n_tokens, "encoder": encoder, "decoder": decoder}


# ── Aggregation ───────────────────────────────────────────────────────────────


def aggregate(per_image: list[dict]) -> dict:
    """Compute mean ± std per layer across all images."""
    enc_keys = [k for k in per_image[0]["encoder"] if k != "total"]
    dec_keys = list(per_image[0]["decoder"]["layers"].keys())

    def _ms(vals: list[float]) -> dict:
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return {"mean": round(m, 3), "std": round(s, 3)}

    enc_summary: dict = {}
    enc_totals = [img["encoder"]["total"] for img in per_image]
    total_mean = statistics.mean(enc_totals)
    for key in enc_keys:
        vals = [img["encoder"][key] for img in per_image]
        d = _ms(vals)
        d["pct"] = round(100.0 * d["mean"] / total_mean, 1) if total_mean > 0 else 0.0
        enc_summary[key] = d
    enc_summary["total"] = _ms(enc_totals)

    dec_summary: dict = {}
    dec_totals = [img["decoder"]["total_ms_per_token"] for img in per_image]
    total_mpt_mean = statistics.mean(dec_totals)
    for key in dec_keys:
        vals = [img["decoder"]["layers"][key]["ms_per_token"] for img in per_image]
        d = _ms(vals)
        d["pct"] = round(100.0 * d["mean"] / total_mpt_mean, 1) if total_mpt_mean > 0 else 0.0
        dec_summary[key] = d
    dec_summary["total"] = _ms(dec_totals)

    n_tokens_vals = [img["n_tokens"] for img in per_image]
    return {
        "n_images": len(per_image),
        "n_tokens": {
            "mean": round(statistics.mean(n_tokens_vals), 1),
            "std": round(statistics.stdev(n_tokens_vals) if len(n_tokens_vals) > 1 else 0.0, 1),
            "min": min(n_tokens_vals),
            "max": max(n_tokens_vals),
        },
        "encoder": enc_summary,
        "decoder": dec_summary,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────


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


def print_summary(summary: dict) -> None:
    enc = summary["encoder"]
    dec = summary["decoder"]
    n = summary["n_images"]
    tok = summary["n_tokens"]
    enc_total = enc["total"]["mean"]
    dec_total = dec["total"]["mean"]
    col = 14

    print()
    print("=" * 55)
    print(f"  Encoder  (mean {enc_total:.1f} ms  over {n} images)")
    print("  " + "-" * 51)
    # Sort stages by mean descending
    enc_keys = [k for k in enc if k != "total"]
    for key in sorted(enc_keys, key=lambda k: enc[k]["mean"], reverse=True):
        d = enc[key]
        print(f"  {key:<{col}} {d['mean']:7.1f} ms   {d['pct']:5.1f}%  ±{d['std']:.1f}")
    print()
    print(f"  Decoder  (mean {dec_total:.2f} ms/tok  |  {tok['mean']:.0f} ± {tok['std']:.0f} tokens)")
    print("  " + "-" * 51)
    dec_keys = [k for k in dec if k != "total"]
    for key in sorted(dec_keys, key=lambda k: dec[k]["mean"], reverse=True):
        d = dec[key]
        print(f"  {key:<{col}} {d['mean']:6.3f} ms/tok  {d['pct']:5.1f}%  ±{d['std']:.3f}")
    print("=" * 55)


def save_results(path: str, args, input_shape: list[int], per_image: list[dict], summary: dict) -> None:
    data = {
        "schema_version": 2,
        "script": "bench_layers.py",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "args": {
            "model": args.model,
            "device": args.device,
            "warmup": args.warmup,
            "n_images": args.n_images,
            "pool": args.pool,
            "dataset": getattr(args, "dataset", None),
            "dataset_split": getattr(args, "dataset_split", None),
            "image_dir": args.image_dir,
            "no_patch": args.no_patch,
        },
        "system": _system_info(),
        "input_shape": input_shape,
        "per_image": per_image,
        "summary": summary,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Donut layer profiler (CUDA only, dataset-driven)")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--n-images", type=int, default=10, dest="n_images",
                        help="number of images to profile (default: 10)")
    parser.add_argument("--pool", type=int, default=20,
                        help="size of image pool to sample from")

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--dataset", default=DEFAULT_DATASET)
    src.add_argument("--image-dir", default=None, dest="image_dir")

    parser.add_argument("--dataset-split", default="test", dest="dataset_split")
    parser.add_argument("--image-column", default="image", dest="image_column")
    parser.add_argument("--no-patch", action="store_true")
    parser.add_argument("--save", default=None, metavar="PATH")
    args = parser.parse_args()

    dev = args.device
    if dev != "cuda":
        sys.exit("bench_layers.py requires CUDA (CUDA events needed for per-layer timing)")

    print(f"  GPU : {torch.cuda.get_device_name(0)}")

    print(f"Loading model ({args.model})...")
    processor = DonutProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.to(dev).eval()

    if args.no_patch:
        pass
    else:
        patch_attn_mask_gpu(model)

    print("Loading image pool...")
    pool = load_pool(args)
    n_images = min(args.n_images, len(pool))
    images = random.sample(pool, n_images)

    # Determine input shape from first image
    sample_pv = processor(images[0], return_tensors="pt").pixel_values.to(dev).to(model.dtype)
    input_shape = list(sample_pv.shape)
    del sample_pv

    # Warmup (unhooked)
    print(f"Warmup ({args.warmup} run(s))...")
    pv_warmup = processor(images[0], return_tensors="pt").pixel_values.to(dev).to(model.dtype)
    did_warmup = processor.tokenizer(TASK_PROMPT, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    for _ in range(args.warmup):
        run_pass(model, processor, pv_warmup, did_warmup)
    cuda_sync()

    # Register hooks once (reuse across images, reset between images)
    timers, handles = register_timers(model)

    # Profiling loop
    per_image: list[dict] = []
    for i, img in enumerate(images):
        print(f"  profiling image {i + 1}/{n_images}...", end="\r")
        result = profile_one_image(model, processor, img, dev, timers)
        per_image.append(result)

    remove_handles(handles)
    print(f"  profiled {n_images} images          ")

    summary = aggregate(per_image)
    print_summary(summary)

    if args.save:
        save_results(args.save, args, input_shape, per_image, summary)


if __name__ == "__main__":
    main()
