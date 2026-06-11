"""Speed benchmarking helpers using synthetic tensors.

No real images or dataset downloads required. All functions return dicts
suitable for JSON serialization or pandas. The backend sweep lives in
scripts/bench_speed.py.
"""

import time

import torch

from donut.synthetic import make_decoder_input_ids, make_pixel_values


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_fn(fn, n_warmup: int, n_runs: int) -> dict:
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
    std = (sum((t - mean) ** 2 for t in times_ms) / n) ** 0.5

    return {
        "mean_ms": round(mean, 3),
        "std_ms": round(std, 3),
        "p50_ms": round(times_ms_sorted[n // 2], 3),
        "p95_ms": round(times_ms_sorted[min(n - 1, int(n * 0.95))], 3),
        "n_runs": n,
    }


def bench_encoder(
    model,
    *,
    batch_size: int = 1,
    n_warmup: int = 3,
    n_runs: int = 20,
    seed: int = 42,
) -> dict:
    """Benchmark the encoder forward pass only, on synthetic pixel_values."""
    pixel_values = make_pixel_values(model, batch_size=batch_size, seed=seed)

    def fn():
        with torch.no_grad():
            model.encoder(pixel_values, return_dict=True)

    return time_fn(fn, n_warmup, n_runs)


def bench_generate(
    model,
    *,
    batch_size: int = 1,
    max_new_tokens: int = 20,
    n_warmup: int = 2,
    n_runs: int = 10,
    seed: int = 42,
) -> dict:
    """Benchmark a full generate() call (encoder + cached decoding), synthetic input."""
    pixel_values = make_pixel_values(model, batch_size=batch_size, seed=seed)
    decoder_input_ids = make_decoder_input_ids(model, batch_size=batch_size)
    pad_id = model.config.decoder.pad_token_id
    if pad_id is None:
        pad_id = model.config.decoder.eos_token_id

    def fn():
        with torch.no_grad():
            model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                use_cache=True,
            )

    return time_fn(fn, n_warmup, n_runs)
