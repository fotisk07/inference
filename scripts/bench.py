"""Donut inference benchmark.

Measures encode, decode, and end-to-end latency over N runs and prints
mean ± std. Designed as the canonical before/after comparison tool for
optimization work (flash attention, quantization, speculative decoding, etc).

Usage:
    uv run scripts/bench.py [--warmup 2] [--runs 10] [--no-patch] [--compile] [--device cuda]
"""

from __future__ import annotations

import datetime
import statistics
import sys

import torch
import transformers
from PIL import Image
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from inference.constants import TASK_PROMPT
from inference.model import apply_patch, load_model
from inference.saving import atomic_save_json
from inference.settings import BenchSettings
from inference.stats import fmt, system_info
from inference.timing import CudaTimer

TEST_IMAGE = "test_data/test_data.jpg"


class Settings(BenchSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True, env_prefix="BENCH_", cli_prog_name="bench"
    )
    runs: int = Field(default=10, description="Number of measurement runs")
    compile: bool = Field(default=False, description="torch.compile the encoder")


def run_once(model, processor, pixel_values, decoder_input_ids):
    """One full encode+decode pass. Returns (encode_ms, decode_ms, n_tokens)."""
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


def build_results(
    cfg: Settings, pixel_values, enc_times, dec_times, e2e_times, token_counts, peak_mb
) -> dict:
    def _stat(vals):
        return {
            "mean": round(statistics.mean(vals), 3),
            "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 3),
        }

    return {
        "schema_version": 1,
        "script": "bench.py",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "args": {
            "model": cfg.model,
            "device": cfg.device,
            "warmup": cfg.warmup,
            "runs": cfg.runs,
            "no_patch": cfg.no_patch,
            "compile": cfg.compile,
        },
        "system": system_info(),
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


def main():
    cfg = Settings()
    dev = cfg.device

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

    print("\nLoading model...")
    model, processor = load_model(cfg.model, dev)
    print(f"  dtype       : {next(model.encoder.parameters()).dtype}")
    print(f"  patch       : {apply_patch(model, dev, cfg.no_patch)}")

    if cfg.compile:
        print("  compiling encoder with torch.compile(dynamic=True)...")
        model.encoder = torch.compile(model.encoder, dynamic=True)

    image = Image.open(TEST_IMAGE).convert("RGB")
    pixel_values = (
        processor(image, return_tensors="pt").pixel_values.to(dev).to(model.dtype)
    )
    decoder_input_ids = processor.tokenizer(
        TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(dev)
    print(f"\n  input shape : {list(pixel_values.shape)}")

    print(f"\nWarmup ({cfg.warmup} run(s))...")
    for _ in range(cfg.warmup):
        run_once(model, processor, pixel_values, decoder_input_ids)

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    enc_times, dec_times, e2e_times, token_counts = [], [], [], []

    for _ in range(cfg.runs):
        t_total = CudaTimer()
        t_total.start()
        enc_ms, dec_ms, n_tok = run_once(
            model, processor, pixel_values, decoder_input_ids
        )
        e2e_ms = t_total.stop()

        enc_times.append(enc_ms)
        dec_times.append(dec_ms)
        e2e_times.append(e2e_ms)
        token_counts.append(n_tok)

        if cfg.save:
            peak_mb = (
                torch.cuda.max_memory_allocated() / 1024**2 if dev == "cuda" else 0.0
            )
            atomic_save_json(
                cfg.save,
                build_results(
                    cfg,
                    pixel_values,
                    enc_times,
                    dec_times,
                    e2e_times,
                    token_counts,
                    peak_mb,
                ),
            )

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if dev == "cuda" else 0.0
    mean_tokens = statistics.mean(token_counts)
    tok_per_sec = mean_tokens / (statistics.mean(dec_times) / 1000.0)

    w = 18
    print(f"\nN={cfg.runs} runs")
    print(f"  {'encode':<{w}} : {fmt(enc_times)}")
    print(f"  {'decode':<{w}} : {fmt(dec_times)}")
    print(f"  {'end-to-end':<{w}} : {fmt(e2e_times)}")
    print()
    print(f"  {'tokens/sec':<{w}} : {tok_per_sec:7.1f}")
    print(f"  {'tokens (mean)':<{w}} : {mean_tokens:7.0f}")
    if dev == "cuda":
        print(f"  {'peak GPU mem':<{w}} : {peak_mb:7.0f} MB")


if __name__ == "__main__":
    main()
