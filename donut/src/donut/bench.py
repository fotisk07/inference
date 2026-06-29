"""Speed benchmarking helpers using synthetic tensors.

No real images or dataset downloads required. All functions return dicts
suitable for JSON serialization or pandas. The two atomic units are
bench_infer_step (one generate() call) and bench_train_step (one fwd+bwd+opt
step); both report the same harmonized docs/s metrics (see README.md Metrics). The grid
sweep / per-backend loop over them lives in scripts/.
"""

import time

import torch

from donut.accel import apply_accel, check_accel, revert_accel
from donut.model import (
    autocast,
    decoder_start_ids,
    decoder_vocab_size,
    init_shift_tokens_from_decoder,
    pad_token_id,
    set_encoder_image_size,
)
from donut.synthetic import make_pixel_values


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


def bench_infer_step(
    model,
    *,
    backend: str,
    h: int,
    w: int,
    batch_size: int,
    max_new_tokens: int,
    n_warmup: int = 3,
    n_runs: int = 10,
    seed: int = 42,
) -> dict:
    """Benchmark one inference step (encoder forward + autoregressive decode).

    Harmonized speed metric (see README.md Metrics): docs/s = batch_size / Δt, where one
    doc = one image + its generated token sequence. compute_docs_s and
    encoder_docs_s mean exactly what they mean in the training bench -- the only
    difference is that here the "step" is a generate() call, not fwd+bwd+opt.
    """
    set_encoder_image_size(model, h, w)
    config = {
        "backend": backend,
        "image_height": h,
        "image_width": w,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
    }
    try:
        init_shift_tokens_from_decoder(model)
        apply_accel(model, backend)
        check_accel(model, backend)

        pixel_values = make_pixel_values(model, batch_size=batch_size, seed=seed)
        decoder_input_ids = decoder_start_ids(model, batch_size=batch_size)
        pad_id = pad_token_id(model)

        # Untimed probe (bs=1) to read the live token counts from the model:
        # num_patches is the Swin input grid the encoder ingests; num_image_tokens
        # is the encoder output seq len -- the KV length the decoder cross-attends
        # to. Both read from actual forwards, not an analytic formula.
        probe = make_pixel_values(model, batch_size=1, seed=seed)
        with torch.no_grad():
            patch_emb, _ = model.encoder.embeddings.patch_embeddings(probe)
            enc_probe = model.encoder(probe, return_dict=True)
        config["num_patches"] = patch_emb.shape[1]
        config["num_image_tokens"] = enc_probe.last_hidden_state.shape[1]

        def encoder_fwd():
            with torch.no_grad():
                model.encoder(pixel_values, return_dict=True)

        def generate_full():
            with torch.no_grad():
                return model.generate(
                    pixel_values=pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=max_new_tokens,
                    pad_token_id=pad_id,
                    use_cache=True,
                )

        enc = time_fn(encoder_fwd, n_warmup, n_runs, verbose=False)
        full = time_fn(generate_full, n_warmup, n_runs, verbose=False)

        encoder_ms = enc["mean_ms"]
        total_ms = full["mean_ms"]
        return {
            **config,
            "status": "ok",
            "new_tokens": float(max_new_tokens),
            "encoder_fwd_ms": round(encoder_ms, 3),
            "decode_ms": round(max(total_ms - encoder_ms, 0.0), 3),
            "total_ms": round(total_ms, 3),
            "total_p50_ms": full["p50_ms"],
            "total_p95_ms": full["p95_ms"],
            "compute_docs_s": round(batch_size / (total_ms / 1000), 2),
            "encoder_docs_s": round(batch_size / (encoder_ms / 1000), 2),
            "peak_mem_mb": _peak_mem_mb(generate_full),
        }
    except Exception as e:
        return {**config, "status": "error", "error": str(e)}
    finally:
        revert_accel(model)


def bench_train_step(
    model,
    *,
    backend: str,
    h: int,
    w: int,
    batch_size: int,
    max_length: int,
    precision: str = "bf16",
    grad_clip: float = 1.0,
    n_warmup: int = 3,
    n_runs: int = 10,
    seed: int = 42,
) -> dict:
    """Benchmark one training step for one (backend, size, batch, max_length) combo.

    The timed step mirrors the real train.py update: forward → backward → grad-clip →
    optimizer.step. docs/s = batch_size / Δt, doc = image + its label sequence; see
    README.md (Metrics).
    """
    set_encoder_image_size(model, h, w)
    config = {
        "backend": backend,
        "image_height": h,
        "image_width": w,
        "batch_size": batch_size,
        "max_length": max_length,
        "grad_clip": grad_clip,
    }
    try:
        init_shift_tokens_from_decoder(model)
        apply_accel(model, backend)
        check_accel(model, backend)

        param_device = next(model.parameters()).device
        device = "cuda" if param_device.type == "cuda" else "cpu"
        pixel_values = make_pixel_values(model, batch_size=batch_size, seed=seed)
        vocab = decoder_vocab_size(model)
        gen = torch.Generator(device=param_device).manual_seed(seed)
        labels = torch.randint(
            0, vocab, (batch_size, max_length), device=param_device, generator=gen
        )

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-9)

        def encoder_fwd():
            with torch.no_grad(), autocast(device, precision):
                model.encoder(pixel_values)

        def forward():
            with torch.no_grad(), autocast(device, precision):
                model(pixel_values=pixel_values, labels=labels).loss

        def forward_backward():
            optimizer.zero_grad()
            with autocast(device, precision):
                loss = model(pixel_values=pixel_values, labels=labels).loss
            loss.backward()

        def full_step():
            optimizer.zero_grad()
            with autocast(device, precision):
                loss = model(pixel_values=pixel_values, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        enc = time_fn(encoder_fwd, n_warmup, n_runs, verbose=False)
        fwd = time_fn(forward, n_warmup, n_runs, verbose=False)
        fb = time_fn(forward_backward, n_warmup, n_runs, verbose=False)
        full = time_fn(full_step, n_warmup, n_runs, verbose=False)

        encoder_ms = enc["mean_ms"]
        forward_ms = fwd["mean_ms"]
        total_ms = full["mean_ms"]
        return {
            **config,
            "status": "ok",
            "encoder_fwd_ms": round(encoder_ms, 3),
            "decoder_fwd_ms": round(max(forward_ms - encoder_ms, 0.0), 3),
            "forward_ms": round(forward_ms, 3),
            "backward_ms": round(max(fb["mean_ms"] - forward_ms, 0.0), 3),
            "optim_ms": round(max(full["mean_ms"] - fb["mean_ms"], 0.0), 3),
            "total_ms": round(total_ms, 3),
            "total_p50_ms": full["p50_ms"],
            "total_p95_ms": full["p95_ms"],
            "compute_docs_s": round(batch_size / (total_ms / 1000), 2),
            "encoder_docs_s": round(batch_size / (encoder_ms / 1000), 2),
            "peak_mem_mb": _peak_mem_mb(full_step),
        }
    except Exception as e:
        return {**config, "status": "error", "error": str(e)}
    finally:
        revert_accel(model)
