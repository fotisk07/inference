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
from pathlib import Path

import torch
import typer
from prettytable import PrettyTable
from tqdm import tqdm

from donut.bench import bench_train_step
from donut.constants import (
    DEFAULT_IMAGE_SIZE_STR,
    DEFAULT_MAX_LENGTH,
    MODEL_ID,
    GLOBAL_OUT_DIR,
)
from donut.model import load_baseline_model
from donut.runio import parse_image_sizes, parse_ints, run_meta, save_record


def _filename(backend: str, h: int, w: int, batch_size: int, max_length: int) -> str:
    return f"{backend}__{h}x{w}__bs{batch_size}__ml{max_length}.json"


app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_id: str = MODEL_ID,
    device: str | None = None,
    # Training-specific: fp32 master weights, bf16 compute via autocast (vs the
    # inference bench's --dtype sweep). "bf16" = autocast on CUDA; "fp32" = off.
    precision: str = "bf16",
    seed: int = 42,
    out: Path = GLOBAL_OUT_DIR / "results" / "bench_train",
    tiny: bool = False,
    backends: str = "baseline,eager,sdpa,fa",
    image_sizes: str = DEFAULT_IMAGE_SIZE_STR,
    batch_sizes: str = "1",
    max_length: int = DEFAULT_MAX_LENGTH,
    n_runs: int = 10,
    n_warmup: int = 3,
    force: bool = False,
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


if __name__ == "__main__":
    app()
