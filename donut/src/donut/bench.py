"""Speed benchmarking helpers using synthetic tensors.

No real images or dataset downloads required. All functions return dicts
suitable for JSON serialization or pandas. bench_one_config is the atomic
per-config unit; the grid sweep over configs lives in scripts/bench_speed.py.
"""

import time

import torch

from donut.accel import apply_accel, check_accel, revert_accel
from donut.synthetic import make_decoder_input_ids, make_pixel_values


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_mem_mb(fn) -> float | None:
    """Peak CUDA memory (MB) during one fn() call; None on CPU.

    Resets the peak counter so the measurement reflects this call's high-water
    mark (model weights are already resident, so it captures weights +
    activations for this config). Run after warmup so allocator caching has
    settled.
    """
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return round(torch.cuda.max_memory_allocated() / 1024**2, 1)


def time_fn(fn, n_warmup: int, n_runs: int, verbose=True) -> dict:
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

    res = {
        "mean_ms": round(mean, 3),
        "std_ms": round(std, 3),
        "p50_ms": round(times_ms_sorted[n // 2], 3),
        "p95_ms": round(times_ms_sorted[min(n - 1, int(n * 0.95))], 3),
        "n_runs": n,
    }

    if verbose:
        res["times"] = times_ms

    return res


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

    stats = time_fn(fn, n_warmup, n_runs)
    stats["images_per_s"] = round(1000 / stats["mean_ms"] * batch_size, 3)
    stats["peak_mem_mb"] = _peak_mem_mb(fn)

    return stats


def bench_generate(
    model,
    *,
    batch_size: int = 1,
    max_new_tokens: int = 20,
    gen_mode: str = "fixed",
    n_warmup: int = 2,
    n_runs: int = 10,
    seed: int = 42,
) -> dict:
    """Benchmark a full generate() call (encoder + cached decoding), synthetic input.

    gen_mode controls how many tokens are decoded:
      "fixed" -- always emit exactly max_new_tokens (min == max). Clean per-step
                 timing, decoupled from content; the default.
      "eos"   -- stop naturally at EOS (capped by max_new_tokens), so latency
                 reflects content-dependent decode length. With synthetic pixels
                 the stopping point is model-dependent noise, so this is most
                 meaningful when max_new_tokens is set to a representative real
                 output length. The realized mean new-token count is reported as
                 "new_tokens" and used for throughput.
    """
    pixel_values = make_pixel_values(model, batch_size=batch_size, seed=seed)
    decoder_input_ids = make_decoder_input_ids(model, batch_size=batch_size)
    prompt_len = decoder_input_ids.shape[1]
    pad_id = model.config.decoder.pad_token_id
    if pad_id is None:
        pad_id = model.config.decoder.eos_token_id
    min_new_tokens = max_new_tokens if gen_mode == "fixed" else 1

    def fn():
        with torch.no_grad():
            return model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                pad_token_id=pad_id,
                use_cache=True,
            )

    # Realized new-token count: exact in fixed mode; measured once in eos mode.
    if gen_mode == "fixed":
        new_tokens = float(max_new_tokens)
    else:
        out = fn()
        new_tokens = round(out.shape[1] - prompt_len, 2)

    stats = time_fn(fn, n_warmup, n_runs)
    stats["new_tokens"] = new_tokens
    stats["tokens_per_s"] = round(1000 / stats["mean_ms"] * new_tokens * batch_size, 3)
    stats["peak_mem_mb"] = _peak_mem_mb(fn)

    return stats


def bench_one_config(
    model,
    *,
    backend: str,
    h: int,
    w: int,
    batch_size: int,
    max_new_tokens: int,
    gen_mode: str = "fixed",
    n_runs: int = 10,
    n_warmup: int = 3,
    seed: int = 42,
) -> dict:
    """Benchmark exactly one (backend, size, batch, max_new_tokens) combo.

    Self-contained: applies the backend, runs encoder + generate timing, and
    always reverts before returning -- even on failure -- so a caller can loop
    this over many configs on one shared model without state leaking between
    iterations. Errors are caught and reported as a "error" status record
    instead of raising, so one bad config doesn't abort a whole sweep.
    """
    model.encoder.config.image_size = [h, w]
    config = {
        "backend": backend,
        "image_height": h,
        "image_width": w,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "gen_mode": gen_mode,
    }
    try:
        apply_accel(model, backend)
        check_accel(model, backend)
        encoder = bench_encoder(
            model, batch_size=batch_size, n_warmup=n_warmup, n_runs=n_runs, seed=seed
        )
        generate = bench_generate(
            model,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            gen_mode=gen_mode,
            n_warmup=n_warmup,
            n_runs=n_runs,
            seed=seed,
        )
        return {**config, "status": "ok", "encoder": encoder, "generate": generate}
    except Exception as e:
        return {
            **config,
            "status": "error",
            "error": str(e),
            "encoder": None,
            "generate": None,
        }
    finally:
        revert_accel(model)
