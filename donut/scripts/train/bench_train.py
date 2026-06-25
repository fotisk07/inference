"""Benchmark a single training step across donut accel backends.

Answers the mechanistic half of "do the accelerations speed up training": with the
dataloader removed (one fixed in-memory batch, reused), how long does a training step
take per backend, and in which component does the time go? Mirrors the inference bench
(scripts/inference/bench_speed.py) — same grid (backends × image sizes × batch sizes),
same harmonized docs/s metric. The atomic per-combo timer is donut.bench.
bench_train_step (twin of bench_infer_step); this CLI just sweeps it and tabulates.

Metrics (see METRICS.md): docs/s = batch_size / Δt.
  compute docs/s   — Δt = one full fwd+bwd+opt step (no data loading), GPU-synced.
  encoder docs/s   — Δt = encoder forward only (isolates the Swin SDPA patch).
Component ms/step: encoder_fwd, decoder_fwd, backward, optim_step.
"""

import itertools
import time
from pathlib import Path

import torch
import typer
from prettytable import PrettyTable
from torch.utils.data import DataLoader

from donut.bench import bench_train_step
from donut.constants import MODEL_ID
from donut.dataset import DonutDataset, load_samples
from donut.model import load_baseline_model, load_model
from donut.runio import parse_image_sizes, parse_ints

app = typer.Typer(add_completion=False)


def _dataloader_probe(
    processor,
    data_json: str,
    batch_size: int,
    num_workers: int,
    n_batches: int,
) -> dict:
    """Real-data loading throughput — the practical bottleneck the compute bench hides.

    Compare its docs/s to the compute docs/s above: if loading is much slower, real
    training is data-bound and the backend choice can't move the wall clock.
    """
    samples = load_samples(Path(data_json))
    ds = DonutDataset(samples, processor, max_length=128)
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
    model_name: str = MODEL_ID,
    image_sizes: str = "1280x960",
    batch_sizes: str = "1",
    max_length: int = 128,
    precision: str = "bf16",
    n_warmup: int = 3,
    n_runs: int = 10,
    seed: int = 42,
    # Tiny offline model on CPU — proves the harness without downloads or a GPU.
    tiny: bool = False,
    # If given, also probe real dataloader throughput (the practical bottleneck).
    data_json: str | None = None,
    num_workers: int = 4,
    device: str | None = None,
) -> None:
    """Per-backend training-step timing breakdown (compute-only, dataloader removed)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    backend_list = [b.strip() for b in backends.split(",") if b.strip()]
    image_sizes_list = parse_image_sizes(image_sizes)
    batch_sizes_list = parse_ints(batch_sizes)

    # baseline load (no accel); each backend is applied/reverted per combo.
    model, _ = load_baseline_model(model_name, device, torch.float32, tiny=tiny)

    print(
        f"\nTraining-step bench  device={device}  precision={precision}  ml={max_length}\n"
    )

    records = [
        bench_train_step(
            model,
            backend=backend,
            h=h,
            w=w,
            batch_size=bs,
            max_length=max_length,
            precision=precision,
            n_warmup=n_warmup,
            n_runs=n_runs,
            seed=seed,
        )
        for backend, (h, w), bs in itertools.product(
            backend_list, image_sizes_list, batch_sizes_list
        )
    ]

    table = PrettyTable()
    table.field_names = [
        "size",
        "backend",
        "bs",
        "status",
        "en_fwd",
        "dec_fwd",
        "bwd",
        "opt",
        "total",
        "cmp_doc",
        "enc_doc",
        "peak_mb",
    ]
    for r in records:
        row_key = [
            f"{r['image_height']}x{r['image_width']}",
            r["backend"],
            r["batch_size"],
        ]
        if r["status"] == "ok":
            table.add_row(
                [
                    *row_key,
                    r["status"],
                    r["encoder_fwd_ms"],
                    r["decoder_fwd_ms"],
                    r["backward_ms"],
                    r["optim_ms"],
                    r["total_ms"],
                    r["compute_docs_s"],
                    r["encoder_docs_s"],
                    "-" if r["peak_mem_mb"] is None else r["peak_mem_mb"],
                ]
            )
        else:
            table.add_row([*row_key, "ERROR", *["-"] * 8])
    print(table)

    if data_json:
        _, processor = load_model(
            model_id=model_name, device=device, dtype=torch.float32, backend="baseline"
        )
        probe = _dataloader_probe(
            processor,
            data_json,
            batch_sizes_list[0],
            num_workers,
            n_runs,
        )
        probe_table = PrettyTable()
        probe_table.field_names = ["num_workers", "mean_batch_ms", "loader_doc_s"]
        probe_table.add_row(
            [probe["num_workers"], probe["mean_batch_ms"], probe["loader_docs_s"]]
        )
        print(probe_table)


if __name__ == "__main__":
    app()
