import itertools
import json
from pathlib import Path
from typing import Literal

import typer
from prettytable import PrettyTable
from tqdm import tqdm

from _common import load_baseline_model, run_meta, save_record
from donut.bench import bench_one_config
from donut.constants import MODEL_ID


def _parse_ints(s: str) -> list[int]:
    return [int(tok.strip()) for tok in s.split(",") if tok.strip()]


def _parse_image_sizes(s: str) -> list[tuple[int, int]]:
    sizes = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        h_str, w_str = token.split("x")
        sizes.append((int(h_str), int(w_str)))
    return sizes


def _filename(
    backend: str, h: int, w: int, batch_size: int, max_new_tokens: int
) -> str:
    return f"{backend}__{h}x{w}__bs{batch_size}__mnt{max_new_tokens}.json"


app = typer.Typer()


@app.command()
def main(
    model_id: str = MODEL_ID,
    device: str = "cuda",
    dtype: Literal["bf16", "f16", "f32"] = "bf16",
    seed: int = 42,
    out: Path = Path("results/bench_speed"),
    tiny: bool = False,
    backends: str = "baseline,eager,sdpa,sdpa_flash,sdpa_math,sdpa_efficient,sdpa_cudnn,fa",
    image_sizes: str = "1280x960",
    batch_sizes: str = "1",
    max_new_tokens: str = "32",
    gen_mode: Literal["fixed", "eos"] = "fixed",
    n_runs: int = 10,
    n_warmup: int = 3,
    force: bool = False,
) -> None:
    backends_list = [b.strip() for b in backends.split(",") if b.strip()]
    image_sizes_list = _parse_image_sizes(image_sizes)
    batch_sizes_list = _parse_ints(batch_sizes)
    max_new_tokens_list = _parse_ints(max_new_tokens)

    model, model_id = load_baseline_model(model_id, device, dtype, tiny)
    meta = run_meta(device, dtype, model_id)

    combos = list(
        itertools.product(
            backends_list, image_sizes_list, batch_sizes_list, max_new_tokens_list
        )
    )
    records = []
    progress = tqdm(combos, desc="bench grid")
    for backend, (h, w), bs, mnt in progress:
        name = _filename(backend, h, w, bs, mnt)
        progress.set_postfix_str(name)
        path = out / name
        if path.exists() and not force:
            tqdm.write(f"skip (exists): {name}")
            records.append(json.loads(path.read_text()))
            continue

        record = bench_one_config(
            model,
            backend=backend,
            h=h,
            w=w,
            batch_size=bs,
            max_new_tokens=mnt,
            gen_mode=gen_mode,
            n_runs=n_runs,
            n_warmup=n_warmup,
            seed=seed,
        )
        save_record(out, name, {**meta, **record})
        records.append(record)

    table = PrettyTable()
    table.field_names = [
        "size",
        "backend",
        "bs",
        "mnt",
        "status",
        "enc ms",
        "img/s",
        "gen ms",
        "tok/s",
    ]
    for r in records:
        row_key = [
            f"{r['image_height']}x{r['image_width']}",
            r["backend"],
            r["batch_size"],
            r["max_new_tokens"],
        ]
        if r["status"] == "ok":
            table.add_row(
                [
                    *row_key,
                    r["status"],
                    r["encoder"]["mean_ms"],
                    r["encoder"]["images_per_s"],
                    r["generate"]["mean_ms"],
                    r["generate"]["tokens_per_s"],
                ]
            )
        else:
            table.add_row([*row_key, "ERROR", "-", "-", "-", "-"])
    print(table)


if __name__ == "__main__":
    app()
