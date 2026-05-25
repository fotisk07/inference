"""Donut inference benchmark.

Measures encode, decode, and end-to-end latency over N runs and prints
mean ± std. Designed as the canonical before/after comparison tool for
optimization work (flash attention, quantization, speculative decoding, etc).

Usage:
    uv run bench.py [--warmup 2] [--runs 10] [--no-patch] [--compile] [--device cuda]
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import statistics
import sys
import time

import torch
import transformers
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from patches import patch_attn_mask, patch_attn_mask_gpu

MODEL_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"
TASK_PROMPT = "<s_cord-v2>"
TEST_IMAGE = "test_data/test_data.jpg"


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


def run_once(model, processor, pixel_values, decoder_input_ids):
    """One full encode+decode pass. Returns (preprocess_ms, encode_ms, decode_ms, n_tokens)."""
    t = CudaTimer()

    t.start()
    with torch.no_grad():
        enc_out = model.encoder(pixel_values, return_dict=True)
    encode_ms = t.stop()

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

    prompt_len = decoder_input_ids.shape[1]
    row = seqs[0, prompt_len:]
    eos = (row == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    n_tokens = int(eos[0].item()) + 1 if len(eos) > 0 else len(row)

    return encode_ms, decode_ms, n_tokens


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
    pixel_values,
    enc_times: list[float],
    dec_times: list[float],
    e2e_times: list[float],
    token_counts: list[int],
    peak_mb: float,
) -> None:
    def _stat(vals: list[float]) -> dict:
        return {
            "mean": round(statistics.mean(vals), 3),
            "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 3),
        }

    data = {
        "schema_version": 1,
        "script": "bench.py",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "args": {
            "model": args.model,
            "device": args.device,
            "warmup": args.warmup,
            "runs": args.runs,
            "no_patch": args.no_patch,
            "compile": args.compile,
        },
        "system": _system_info(),
        "image": {
            "path": TEST_IMAGE,
            "input_shape": list(pixel_values.shape),
        },
        "runs": {
            "encode_ms": enc_times,
            "decode_ms": dec_times,
            "e2e_ms": e2e_times,
            "n_tokens": token_counts,
        },
        "summary": {
            "encode_ms": _stat(enc_times),
            "decode_ms": _stat(dec_times),
            "e2e_ms": _stat(e2e_times),
            "tokens_per_sec": round(
                statistics.mean(token_counts) / (statistics.mean(dec_times) / 1000.0), 2
            ),
            "tokens_mean": round(statistics.mean(token_counts), 1),
            "peak_gpu_mb": round(peak_mb, 1),
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  stats saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="Donut inference benchmark")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--no-patch", action="store_true", help="skip attn_mask patch")
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile the encoder"
    )
    parser.add_argument(
        "--save", default=None, metavar="PATH", help="save per-run stats to JSON"
    )
    args = parser.parse_args()

    dev = args.device

    # ── System info ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Donut inference benchmark")
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

    if args.compile:
        print("  compiling encoder with torch.compile(dynamic=True)...")
        model.encoder = torch.compile(model.encoder, dynamic=True)

    # ── Preprocess ───────────────────────────────────────────────────────────
    image = Image.open(TEST_IMAGE).convert("RGB")
    pixel_values = (
        processor(image, return_tensors="pt").pixel_values.to(dev).to(model.dtype)
    )
    decoder_input_ids = processor.tokenizer(
        TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(dev)
    print(f"\n  input shape : {list(pixel_values.shape)}")

    # ── Warmup ───────────────────────────────────────────────────────────────
    print(f"\nWarmup ({args.warmup} run(s))...")
    for _ in range(args.warmup):
        with torch.no_grad():
            run_once(model, processor, pixel_values, decoder_input_ids)

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ── Measurement loop ─────────────────────────────────────────────────────
    enc_times, dec_times, e2e_times, token_counts = [], [], [], []

    for _ in range(args.runs):
        t_total = CudaTimer()
        t_total.start()
        with torch.no_grad():
            enc_ms, dec_ms, n_tok = run_once(
                model, processor, pixel_values, decoder_input_ids
            )
        e2e_ms = t_total.stop()

        enc_times.append(enc_ms)
        dec_times.append(dec_ms)
        e2e_times.append(e2e_ms)
        token_counts.append(n_tok)

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if dev == "cuda" else 0.0
    mean_tokens = statistics.mean(token_counts)
    mean_dec_s = statistics.mean(dec_times) / 1000.0
    tok_per_sec = mean_tokens / mean_dec_s

    # ── Results ──────────────────────────────────────────────────────────────
    w = 18
    print(f"\nN={args.runs} runs")
    print(f"  {'encode':<{w}} : {fmt(enc_times)}")
    print(f"  {'decode':<{w}} : {fmt(dec_times)}")
    print(f"  {'end-to-end':<{w}} : {fmt(e2e_times)}")
    print()
    print(f"  {'tokens/sec':<{w}} : {tok_per_sec:7.1f}")
    print(f"  {'tokens (mean)':<{w}} : {mean_tokens:7.0f}")
    if dev == "cuda":
        print(f"  {'peak GPU mem':<{w}} : {peak_mb:7.0f} MB")

    if args.save:
        save_stats(
            args.save,
            args,
            pixel_values,
            enc_times,
            dec_times,
            e2e_times,
            token_counts,
            peak_mb,
        )


if __name__ == "__main__":
    main()
