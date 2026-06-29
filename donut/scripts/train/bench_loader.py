"""Benchmark real-data dataloader throughput across worker counts.

Answers the practical half of "do the accelerations speed up training": the
compute bench (scripts/train/bench_train.py) removes the dataloader to isolate
the step; this measures the dataloader on its own. Compare its docs/s to
bench_train's compute docs/s — if loading is slower, real training is data-bound
and the backend choice can't move the wall clock.

Mirrors the other bench CLIs (bench_train, bench_speed): same defaults from
constants, same GLOBAL_OUT_DIR out path, same per-combo JSON records + table.
The per-combo timer is inline here (no donut.bench twin) — the loop is trivial.

Metric (see README.md Metrics): loader docs/s = batch_size / mean_batch_Δt.
  Δt = wall time the DataLoader takes to produce one ready batch (decode +
  processor resize, parallelised across num_workers).
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
from transformers import DonutProcessor

from donut.constants import (
    DEFAULT_IMAGE_SIZE_STR,
    DEFAULT_MAX_LENGTH,
    MODEL_ID,
    GLOBAL_OUT_DIR,
)
from donut.dataset import DonutDataset, load_samples
from donut.model import set_processor_image_size
from donut.runio import parse_image_sizes, parse_ints, run_meta, save_record


def _filename(num_workers: int, h: int, w: int, batch_size: int) -> str:
    return f"nw{num_workers}__{h}x{w}__bs{batch_size}.json"


def bench_loader_step(
    processor: DonutProcessor,
    samples: list[dict],
    *,
    h: int,
    w: int,
    batch_size: int,
    num_workers: int,
    max_length: int,
    n_runs: int,
    n_warmup: int,
    seed: int,
) -> dict:
    """Time the mean wall clock to produce one ready batch from real data."""
    set_processor_image_size(processor, h, w)
    ds = DonutDataset(samples, processor, max_length=max_length)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )

    times_ms: list[float] = []
    end = time.perf_counter()
    for i, _ in enumerate(loader):
        dt = (time.perf_counter() - end) * 1000
        if i >= n_warmup:  # skip worker spin-up / first-touch caching
            times_ms.append(dt)
        end = time.perf_counter()
        if len(times_ms) >= n_runs:
            break

    if not times_ms:
        return _result(h, w, batch_size, num_workers, status="too_few_batches")
    mean_ms = sum(times_ms) / len(times_ms)
    return _result(
        h,
        w,
        batch_size,
        num_workers,
        status="ok",
        mean_batch_ms=round(mean_ms, 3),
        loader_docs_s=round(batch_size / (mean_ms / 1000), 2),
        n_batches=len(times_ms),
    )


def _result(h, w, batch_size, num_workers, *, status, **rest) -> dict:
    return {
        "image_height": h,
        "image_width": w,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "status": status,
        **rest,
    }


app = typer.Typer(add_completion=False)


@app.command()
def main(
    data_json: str,
    model_id: str = MODEL_ID,
    seed: int = 42,
    out: Path = GLOBAL_OUT_DIR / "results" / "bench_loader",
    image_sizes: str = DEFAULT_IMAGE_SIZE_STR,
    batch_sizes: str = "1",
    num_workers: str = "0,4",
    max_length: int = DEFAULT_MAX_LENGTH,
    n_runs: int = 10,
    n_warmup: int = 3,
    force: bool = False,
) -> None:
    """Per-(workers × size × batch) dataloader throughput on real data."""
    image_sizes_list = parse_image_sizes(image_sizes)
    batch_sizes_list = parse_ints(batch_sizes)
    num_workers_list = parse_ints(num_workers)

    processor = DonutProcessor.from_pretrained(model_id)
    samples = load_samples(Path(data_json))
    meta = run_meta(None, None, model_id)

    combos = list(
        itertools.product(num_workers_list, image_sizes_list, batch_sizes_list)
    )
    records = []
    progress = tqdm(combos, desc="loader grid")
    for nw, (h, w), bs in progress:
        name = _filename(nw, h, w, bs)
        progress.set_postfix_str(name)
        path = out / name
        if path.exists() and not force:
            tqdm.write(f"skip (exists): {name}")
            records.append(json.loads(path.read_text()))
            continue

        record = bench_loader_step(
            processor,
            samples,
            h=h,
            w=w,
            batch_size=bs,
            num_workers=nw,
            max_length=max_length,
            n_runs=n_runs,
            n_warmup=n_warmup,
            seed=seed,
        )
        save_record(out, name, {**meta, **record})
        records.append(record)

    table = PrettyTable()
    table.field_names = [
        "size",
        "nw",
        "bs",
        "status",
        "mean_ms",
        "loader_doc_s",
        "n_batches",
    ]
    for r in records:
        row_key = [
            f"{r['image_height']}x{r['image_width']}",
            r["num_workers"],
            r["batch_size"],
        ]
        if r["status"] == "ok":
            table.add_row(
                [
                    *row_key,
                    r["status"],
                    r["mean_batch_ms"],
                    r["loader_docs_s"],
                    r["n_batches"],
                ]
            )
        else:
            table.add_row([*row_key, r["status"], *["-"] * 3])
    print(table)


if __name__ == "__main__":
    app()
