"""Latency/throughput benchmark: baseline vs accelerated backends, synthetic data.

For each (image_size × backend × batch_size) combination the preset is applied,
structurally verified with check_accel (never bench an unverified config),
benchmarked, and fully reverted. A "baseline" row (no optimizations at all, not
even mask caching) is always included as the speedup reference within each
image-size group.

Outputs (one self-describing file per config, so partial/repeated sweeps
accumulate in the same directory — the notebooks glob them back together):
    <out>/bench_<HxW>_<backend>_bs<N>.json   {meta, record}

Each measurement is OOM-safe: on CUDA OOM the cache is freed and a status:"oom"
row is recorded (no timings) so a wide image×batch sweep maps its limits instead
of aborting.

Usage:
    uv run python scripts/bench_speed.py --backends eager,sdpa,fa --batch-sizes 1,2,4
    uv run python scripts/bench_speed.py --image-sizes 640x480,960x720,1280x960
    uv run python scripts/bench_speed.py --tiny --backends eager,sdpa --n-runs 3
"""

import json

import torch
from _common import base_parser, load_baseline_model, run_meta, save_json, save_record

from donut.accel import apply_accel, check_accel, revert_accel
from donut.bench import bench_encoder, bench_generate
from prettytable import PrettyTable


def _parse_image_sizes(s: str) -> list[tuple[int, int]]:
    sizes = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        h_str, w_str = token.split("x")
        sizes.append((int(h_str), int(w_str)))
    return sizes


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--backends", default="eager,sdpa,fa")
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument(
        "--image-sizes",
        default="1280x960",
        help="comma-separated HxW pairs, e.g. 640x480,960x720,1280x960. "
        "H and W must be divisible by 40 (patch_size=4 × window_size=10) for donut-base.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--gen-mode",
        choices=["fixed", "eos"],
        default="fixed",
        help="fixed: always emit max-new-tokens (clean per-step timing). "
        "eos: stop at EOS, capped by max-new-tokens (content-dependent length).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="override runs for encoder AND generate",
    )
    parser.add_argument("--n-warmup", type=int, default=None, help="override warmups")
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    image_sizes = _parse_image_sizes(args.image_sizes)

    model, model_id = load_baseline_model(args)
    meta = run_meta(args, model_id)

    def measure(backend: str, h: int, w: int, bs: int) -> dict:
        """One (backend, batch_size) measurement, OOM-safe.

        On CUDA OOM the GPU cache is freed and a status:"oom" row (no timings) is
        returned, so a wide sweep records the limit and keeps going.
        """
        base = {
            "image_height": h,
            "image_width": w,
            "backend": backend,
            "batch_size": bs,
            "gen_mode": args.gen_mode,
        }
        enc = bench_encoder(
            model,
            batch_size=bs,
            n_warmup=args.n_warmup,
            n_runs=args.n_runs,
            seed=args.seed,
        )
        gen = bench_generate(
            model,
            batch_size=bs,
            max_new_tokens=args.max_new_tokens,
            gen_mode=args.gen_mode,
            n_warmup=args.n_warmup,
            n_runs=args.n_runs,
            seed=args.seed,
        )

        return {
            **base,
            "status": "ok",
            "encoder": enc,
            "generate": gen,
        }

    records = []

    for h, w in image_sizes:
        print(f"\n--- image size {h}x{w} ---")
        # Temporarily override config so make_pixel_values uses the correct shape.
        # Swin handles variable image sizes: relative position bias is per-window
        # (size-agnostic) and shifted-window masks are recomputed from feature-map shape.
        model.encoder.config.image_size = [h, w]

        size_records = [measure("baseline", h, w, bs) for bs in batch_sizes]
        for backend in backends:
            apply_accel(model, backend)
            check_accel(model, backend)
            size_records.extend(measure(backend, h, w, bs) for bs in batch_sizes)
            revert_accel(model)

        records.extend(size_records)

    save_json(
        args.out / "bench_speed.json",
        {"meta": run_meta(args, model_id), "records": records},
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
    main()
