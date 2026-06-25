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
import json
import time
from pathlib import Path

import torch
import typer
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from tqdm import tqdm

from donut.bench import bench_train_step
from donut.constants import DEFAULT_MAX_LENGTH, MODEL_ID
from donut.dataset import DonutDataset, load_samples
from donut.model import load_baseline_model, load_model
from donut.runio import parse_image_sizes, parse_ints, run_meta, save_record


def _filename(backend: str, h: int, w: int, batch_size: int, max_length: int) -> str:
    return f"{backend}__{h}x{w}__bs{batch_size}__ml{max_length}.json"


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
    ds = DonutDataset(samples, processor, max_length=DEFAULT_MAX_LENGTH)
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
    model_id: str = MODEL_ID,
    device: str | None = None,
    # Training-specific: fp32 master weights, bf16 compute via autocast (vs the
    # inference bench's --dtype sweep). "bf16" = autocast on CUDA; "fp32" = off.
    precision: str = "bf16",
    seed: int = 42,
    out: Path = Path("results/bench_train"),
    tiny: bool = False,
    backends: str = "baseline,eager,sdpa,fa",
    image_sizes: str = "1280x960",
    batch_sizes: str = "1",
    max_length: int = DEFAULT_MAX_LENGTH,
    n_runs: int = 10,
    n_warmup: int = 3,
    force: bool = False,
    # Training-specific: if given, also probe real dataloader throughput.
    data_json: str | None = None,
    num_workers: int = 4,
) -> None:
    """Per-backend training-step timing breakdown (compute-only, dataloader removed)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    backends_list = [b.strip() for b in backends.split(",") if b.strip()]
    image_sizes_list = parse_image_sizes(image_sizes)
    batch_sizes_list = parse_ints(batch_sizes)

    # fp32 master weights (no accel); each backend is applied/reverted per combo.
    # bf16 compute is applied at the forward via autocast, controlled by --precision.
    model, model_id = load_baseline_model(model_id, device, torch.float32, tiny=tiny)
    meta = run_meta(device, "f32", model_id)
    meta["precision"] = precision

    print(
        f"\nTraining-step bench  device={device}  precision={precision}  ml={max_length}\n"
    )

    combos = list(itertools.product(backends_list, image_sizes_list, batch_sizes_list))
    records = []
    progress = tqdm(combos, desc="bench grid")
    for backend, (h, w), bs in progress:
        name = _filename(backend, h, w, bs, max_length)
        progress.set_postfix_str(name)
        path = out / name
        if path.exists() and not force:
            tqdm.write(f"skip (exists): {name}")
            records.append(json.loads(path.read_text()))
            continue

        record = bench_train_step(
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
        save_record(out, name, {**meta, **record})
        records.append(record)

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
            model_id=model_id, device=device, dtype=torch.float32, backend="baseline"
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
