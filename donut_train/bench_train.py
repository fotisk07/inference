"""Benchmark a single training step across donut accel backends.

Answers the mechanistic half of "do the accelerations speed up training": with the
dataloader removed (one fixed in-memory batch, reused), how long does a training step
take per backend, and in which component does the time go? Mirrors the inference bench
(donut/scripts/bench_speed.py) and reuses donut.bench.time_fn for warmup + sync + stats.

Metrics (see also train.py): docs/s = batch_size / Δt.
  compute docs/s   — Δt = one full fwd+bwd+opt step (no data loading), GPU-synced.
  encoder docs/s   — Δt = encoder forward only (isolates the Swin SDPA patch).
Component ms/step: encoder_fwd, forward_total, backward (= fwd+bwd − fwd), optim_step.
"""

import json
from pathlib import Path

import torch
import typer
from donut import apply_accel, check_accel, load_model, revert_accel
from donut.bench import _peak_mem_mb, time_fn
from donut.synthetic import make_pixel_values, make_tiny_model

from train import autocast

app = typer.Typer(add_completion=False)


def _attn_state(model) -> dict:
    """Actual attention impls in play — printed so we rely on fact, not theory."""
    block = model.encoder.encoder.layers[0].blocks[0].attention.self
    return {
        "encoder_sdpa_patched": bool(getattr(block, "_sdpa_patched", False)),
        "decoder_attn_impl": model.decoder.config._attn_implementation,
    }


def _bench_backend(
    model,
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: str,
    precision: str,
    batch_size: int,
    n_warmup: int,
    n_runs: int,
) -> dict:
    """Time the components of one training step for the currently-applied backend."""
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
        optimizer.step()

    enc = time_fn(encoder_fwd, n_warmup, n_runs, verbose=False)
    fwd = time_fn(forward, n_warmup, n_runs, verbose=False)
    fb = time_fn(forward_backward, n_warmup, n_runs, verbose=False)
    full = time_fn(full_step, n_warmup, n_runs, verbose=False)

    encoder_ms = enc["mean_ms"]
    forward_ms = fwd["mean_ms"]
    backward_ms = max(fb["mean_ms"] - fwd["mean_ms"], 0.0)
    optim_ms = max(full["mean_ms"] - fb["mean_ms"], 0.0)
    total_ms = full["mean_ms"]

    return {
        "encoder_fwd_ms": round(encoder_ms, 3),
        "decoder_fwd_ms": round(max(forward_ms - encoder_ms, 0.0), 3),
        "forward_ms": round(forward_ms, 3),
        "backward_ms": round(backward_ms, 3),
        "optim_ms": round(optim_ms, 3),
        "total_ms": round(total_ms, 3),
        "total_p50_ms": full["p50_ms"],
        "total_p95_ms": full["p95_ms"],
        "compute_docs_s": round(batch_size / (total_ms / 1000), 2),
        "encoder_docs_s": round(batch_size / (encoder_ms / 1000), 2),
        "peak_mem_mb": _peak_mem_mb(full_step),
    }


def _dataloader_probe(
    data_json: str, batch_size: int, num_workers: int, image_size, n_batches: int
) -> dict:
    """Real-data loading throughput — the practical bottleneck the compute bench hides.

    Compare its docs/s to the compute docs/s above: if loading is much slower, real
    training is data-bound and the backend choice can't move the wall clock.
    """
    import time

    from torch.utils.data import DataLoader

    from dataset import DonutDataset, load_samples
    from transformers import DonutProcessor

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    processor.image_processor.size = {"height": image_size[0], "width": image_size[1]}
    samples = load_samples(Path(data_json))
    ds = DonutDataset(samples, processor, max_length=128, token2json_format=True)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    times_ms = []
    end = time.perf_counter()
    for i, _ in enumerate(loader):
        times_ms.append((time.perf_counter() - end) * 1000)
        end = time.perf_counter()
        if i + 1 >= n_batches:
            break
    mean_ms = sum(times_ms) / len(times_ms)
    return {
        "mean_batch_ms": round(mean_ms, 3),
        "loader_docs_s": round(batch_size / (mean_ms / 1000), 2),
        "n_batches": len(times_ms),
        "num_workers": num_workers,
    }


@app.command()
def main(
    backends: str = "baseline,eager,sdpa,fa",
    model_name: str = "naver-clova-ix/donut-base",
    image_height: int = 1280,
    image_width: int = 960,
    batch_size: int = 4,
    max_length: int = 128,
    precision: str = "bf16",
    n_warmup: int = 3,
    n_runs: int = 10,
    # Tiny offline model on CPU — proves the harness without downloads or a GPU.
    tiny: bool = False,
    # If given, also probe real dataloader throughput (the practical bottleneck).
    data_json: str | None = None,
    num_workers: int = 4,
    device: str | None = None,
    out: str | None = None,
) -> None:
    """Per-backend training-step timing breakdown (compute-only, dataloader removed)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    backend_list = [b.strip() for b in backends.split(",") if b.strip()]

    if tiny:
        model = make_tiny_model().to(device)
    else:
        # baseline load (no accel); each backend is applied/reverted in the loop.
        model, _ = load_model(
            model_id=model_name, device=device, dtype=torch.float32, backend="baseline"
        )
        model.encoder.config.image_size = [image_height, image_width]

    pixel_values = make_pixel_values(model, batch_size=batch_size)
    vocab = model.decoder.config.vocab_size
    labels = torch.randint(0, vocab, (batch_size, max_length), device=device)

    print(f"\nTraining-step bench  device={device}  precision={precision}")
    print(
        f"image={pixel_values.shape[-2]}×{pixel_values.shape[-1]}  batch={batch_size}\n"
    )

    records = []
    for backend in backend_list:
        try:
            apply_accel(model, backend)
            check_accel(model, backend)
            attn = _attn_state(model)
            stats = _bench_backend(
                model,
                pixel_values,
                labels,
                device=device,
                precision=precision,
                batch_size=batch_size,
                n_warmup=n_warmup,
                n_runs=n_runs,
            )
            records.append({"backend": backend, "status": "ok", **attn, **stats})
        except Exception as e:  # noqa: BLE001 — one bad backend shouldn't abort the sweep
            records.append({"backend": backend, "status": "error", "error": str(e)})
        finally:
            revert_accel(model)

    _print_table(records)

    if data_json:
        probe = _dataloader_probe(
            data_json, batch_size, num_workers, (image_height, image_width), n_runs
        )
        print(
            f"\nDataloader (real, {probe['num_workers']} workers): "
            f"{probe['mean_batch_ms']:.1f} ms/batch  →  {probe['loader_docs_s']} docs/s"
        )
        print(
            "Compare to compute docs/s above: if loader docs/s is much lower, real "
            "training is data-bound and the backend can't move the wall clock."
        )

    if out:
        Path(out).write_text(
            json.dumps({"records": records, "device": device}, indent=2)
        )
        print(f"\nSaved → {out}")


def _print_table(records: list[dict]) -> None:
    cols = [
        ("backend", 14),
        ("enc_fwd", 9),
        ("dec_fwd", 9),
        ("bwd", 9),
        ("opt", 8),
        ("total", 9),
        ("cmp_doc/s", 10),
        ("enc_doc/s", 10),
        ("peak_mb", 9),
    ]
    header = "".join(name.rjust(w) for name, w in cols)
    print(header)
    print("-" * len(header))
    for r in records:
        if r["status"] != "ok":
            print(f"{r['backend']:>14}  error: {r['error']}")
            continue
        row = [
            r["backend"],
            f"{r['encoder_fwd_ms']:.2f}",
            f"{r['decoder_fwd_ms']:.2f}",
            f"{r['backward_ms']:.2f}",
            f"{r['optim_ms']:.2f}",
            f"{r['total_ms']:.2f}",
            f"{r['compute_docs_s']:.1f}",
            f"{r['encoder_docs_s']:.1f}",
            "—" if r["peak_mem_mb"] is None else f"{r['peak_mem_mb']:.0f}",
        ]
        print("".join(val.rjust(w) for val, (_, w) in zip(row, cols)))
    print()
    for r in records:
        if r["status"] == "ok":
            print(
                f"  {r['backend']:>14}: encoder_sdpa_patched="
                f"{r['encoder_sdpa_patched']}  decoder_attn={r['decoder_attn_impl']}"
            )


if __name__ == "__main__":
    app()
