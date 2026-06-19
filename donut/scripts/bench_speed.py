from _common import load_baseline_model, run_meta, save_json

from donut.accel import apply_accel, check_accel, revert_accel
from donut.bench import bench_encoder, bench_generate
from prettytable import PrettyTable
from pathlib import Path
import typer
from donut.constants import MODEL_ID
from typing import Literal


def _parse_image_sizes(s: str) -> list[tuple[int, int]]:
    sizes = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        h_str, w_str = token.split("x")
        sizes.append((int(h_str), int(w_str)))
    return sizes


app = typer.Typer()


@app.command()
def main(
    model_id: str = MODEL_ID,
    device: str = "cuda",
    dtype=Literal["bf16", "f16", "f32"],
    seed: int = 42,
    out: Path = Path("results"),
    tiny: bool = False,
    backends: str = "eager,sdpa,fa",
    image_sizes: str = "1280x960",
    batch_sizes: str = "1",
    max_new_tokens: int = 32,
    gen_mode: Literal["fixed", "eos"] = "fixed",
    n_runs: int = 10,
    n_warmup: int = 3,
) -> None:

    batch_sizes = [int(b) for b in batch_sizes.split(",")]
    backends = [b.strip() for b in backends.split(",") if b.strip()]
    image_sizes = _parse_image_sizes(image_sizes)

    model, model_id = load_baseline_model(model_id, device, dtype, tiny)

    def bench_at_size(backend: str, h: int, w: int, bs: int) -> dict:
        rows = []
        for bs in batch_sizes:
            enc = bench_encoder(
                model,
                batch_size=bs,
                n_warmup=n_warmup,
                n_runs=n_runs,
                seed=seed,
            )
            gen = bench_generate(
                model,
                batch_size=bs,
                max_new_tokens=max_new_tokens,
                gen_mode=gen_mode,
                n_warmup=n_warmup,
                n_runs=n_runs,
                seed=seed,
            )

        rows.append(
            {
                "image_height": h,
                "image_width": w,
                "backend": backend,
                "batch_size": bs,
                "gen_mode": gen_mode,
                "status": "ok",
                "encoder": enc,
                "generate": gen,
            }
        )
        return rows

    records = []

    for h, w in image_sizes:
        print(f"\n--- image size {h}x{w} ---")
        # Temporarily override config so make_pixel_values uses the correct shape.
        # Swin handles variable image sizes: relative position bias is per-window
        # (size-agnostic) and shifted-window masks are recomputed from feature-map shape.
        model.encoder.config.image_size = [h, w]

        size_records = bench_at_size("baseline", h=h, w=w)
        for backend in backends:
            apply_accel(model, backend)
            check_accel(model, backend)
            size_records.extend(bench_at_size(backend, h=h, w=w))
            revert_accel(model)

        records.extend(size_records)

    save_json(
        out / "bench_speed.json",
        {"meta": run_meta(device, dtype, model_id), "records": records},
    )
    table = PrettyTable()
    table.field_names = ["size", "backend", "bs", "enc ms", "gen ms", "tok/S"]
    table.add_rows(
        [
            [
                f"{r['image_height']}x{r['image_width']}",
                r["backend"],
                r["batch_size"],
                r["encoder"]["mean_ms"],
                r["generate"]["mean_ms"],
                r["generate"]["mean_ms"],
                r["generate"]["mean tok/s"],
            ]
            for r in records
        ]
    )
    print(table)


if __name__ == "__main__":
    app()
