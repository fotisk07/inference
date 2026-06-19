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
    stats["mean img/s"] = round(stats["mean_ms"] / batch_size, 3)

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
    stats["mean tok/s"] = round(1000 / stats["mean_ms"] * new_tokens * batch_size, 3)

    return stats
