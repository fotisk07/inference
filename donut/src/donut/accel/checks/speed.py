"""Speed benchmarks using synthetic tensors.

No real images or dataset downloads required. All functions return dicts
suitable for direct use in pandas DataFrames or JSON serialization.
Results can be saved to JSON for Jupyter notebook visualization.

Usage:
    from donut import load_model, Backend
    from donut.accel.checks.speed import run_speed_bench

    model, processor = load_model(device="cuda", backend=Backend.EAGER)
    results = run_speed_bench(
        model, processor,
        backends=[Backend.EAGER, Backend.SDPA, Backend.FA2],
        batch_sizes=[1, 2, 4],
        save_path="results/speed.json",
    )
    # pd.DataFrame(r["encoder"] for r in results)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from donut.accel import Backend, apply_accel
from donut.accel.checks.numerical import _make_inputs

if TYPE_CHECKING:
    from transformers import DonutProcessor, VisionEncoderDecoderModel


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_fn(fn, n_warmup: int, n_runs: int) -> dict:
    """Run fn() n_warmup times (discarded), then n_runs times. Return latency stats."""
    for _ in range(n_warmup):
        fn()
    _cuda_sync()

    times_ms = []
    for _ in range(n_runs):
        _cuda_sync()
        t0 = time.perf_counter()
        fn()
        _cuda_sync()
        times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms_sorted = sorted(times_ms)
    n = len(times_ms_sorted)
    mean = sum(times_ms) / n
    variance = sum((t - mean) ** 2 for t in times_ms) / n
    std = variance**0.5

    return {
        "mean_ms": round(mean, 3),
        "std_ms": round(std, 3),
        "p50_ms": round(times_ms_sorted[n // 2], 3),
        "p95_ms": round(times_ms_sorted[int(n * 0.95)], 3),
        "n_runs": n,
    }


def bench_encoder(
    model: VisionEncoderDecoderModel,
    processor: DonutProcessor,
    *,
    batch_size: int = 1,
    n_warmup: int = 3,
    n_runs: int = 20,
    seed: int = 42,
) -> dict:
    """Benchmark encoder forward pass only using synthetic pixel_values."""
    pixel_values, _ = _make_inputs(model, processor, batch_size=batch_size, seed=seed)

    def fn():
        with torch.no_grad():
            model.encoder(pixel_values, return_dict=True)

    return _time_fn(fn, n_warmup, n_runs)


def bench_generate(
    model: VisionEncoderDecoderModel,
    processor: DonutProcessor,
    *,
    batch_size: int = 1,
    max_new_tokens: int = 20,
    n_warmup: int = 2,
    n_runs: int = 10,
    seed: int = 42,
) -> dict:
    """Benchmark full generate() call using synthetic pixel_values."""
    pixel_values, decoder_input_ids = _make_inputs(
        model, processor, batch_size=batch_size, seed=seed
    )

    def fn():
        with torch.no_grad():
            model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
            )

    return _time_fn(fn, n_warmup, n_runs)


def run_speed_bench(
    model: VisionEncoderDecoderModel,
    processor: DonutProcessor,
    backends: list[Backend],
    *,
    batch_sizes: list[int] = [1],
    max_new_tokens: int = 20,
    n_warmup_encoder: int = 3,
    n_runs_encoder: int = 20,
    n_warmup_generate: int = 2,
    n_runs_generate: int = 10,
    save_path: str | None = None,
) -> list[dict]:
    """Benchmark all (backend × batch_size) combinations.

    Re-applies accelerations for each backend in-place. The model is mutated
    progressively — backends must be ordered from least to most aggressive
    (EAGER → SDPA → FA2) to avoid double-patching conflicts, or load a fresh
    model per backend for isolation.

    Each result dict has shape:
        {backend, batch_size, encoder: {mean_ms, ...}, generate: {mean_ms, ...}}

    The returned list is directly convertible to a pandas DataFrame:
        import pandas as pd
        df = pd.json_normalize(results)
    """
    results = []
    for backend in backends:
        apply_accel(model, backend)
        for bs in batch_sizes:
            enc_stats = bench_encoder(
                model,
                processor,
                batch_size=bs,
                n_warmup=n_warmup_encoder,
                n_runs=n_runs_encoder,
            )
            gen_stats = bench_generate(
                model,
                processor,
                batch_size=bs,
                max_new_tokens=max_new_tokens,
                n_warmup=n_warmup_generate,
                n_runs=n_runs_generate,
            )
            results.append(
                {
                    "backend": backend.value
                    if isinstance(backend, Backend)
                    else str(backend),
                    "batch_size": bs,
                    "encoder": enc_stats,
                    "generate": gen_stats,
                }
            )

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, indent=2))

    return results
