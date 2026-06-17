"""Latency/throughput benchmark: baseline vs accelerated backends, synthetic data.

For each (image_size × backend × batch_size) combination the preset is applied,
structurally verified with check_accel (never bench an unverified config),
benchmarked, and fully reverted. A "baseline" row (no optimizations at all, not
even mask caching) is always included as the speedup reference within each
image-size group.

Outputs:
    <out>/bench_speed.json    {meta, records: [...]}, pd.json_normalize-friendly

Usage:
    uv run python scripts/bench_speed.py --backends eager,sdpa,fa --batch-sizes 1,2,4
    uv run python scripts/bench_speed.py --image-sizes 640x480,960x720,1280x960
    uv run python scripts/bench_speed.py --tiny --backends eager,sdpa --n-runs 3
"""

from _common import base_parser, load_baseline_model, run_meta, save_json

from donut.accel import apply_accel, check_accel, fa_available, revert_accel
from donut.bench import bench_encoder, bench_generate


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
        default=None,
        help="override runs for encoder AND generate",
    )
    parser.add_argument("--n-warmup", type=int, default=None, help="override warmups")
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    image_sizes = _parse_image_sizes(args.image_sizes)

    if "fa" in backends and not fa_available():
        print("flash-attn unavailable (needs CUDA + flash-attn) — dropping 'fa'")
        backends = [b for b in backends if b != "fa"]

    n_runs_enc = args.n_runs or 20
    n_runs_gen = args.n_runs or 10
    n_warm_enc = args.n_warmup if args.n_warmup is not None else 3
    n_warm_gen = args.n_warmup if args.n_warmup is not None else 2

    model, model_id = load_baseline_model(args)

    def bench_at_size(backend: str, h: int, w: int) -> list[dict]:
        rows = []
        for bs in batch_sizes:
            enc = bench_encoder(
                model,
                batch_size=bs,
                n_warmup=n_warm_enc,
                n_runs=n_runs_enc,
                seed=args.seed,
            )
            gen = bench_generate(
                model,
                batch_size=bs,
                max_new_tokens=args.max_new_tokens,
                gen_mode=args.gen_mode,
                n_warmup=n_warm_gen,
                n_runs=n_runs_gen,
                seed=args.seed,
            )
            rows.append(
                {
                    "image_height": h,
                    "image_width": w,
                    "backend": backend,
                    "batch_size": bs,
                    "gen_mode": args.gen_mode,
                    "encoder": enc,
                    "generate": gen,
                    "throughput": {
                        "images_per_s": round(1000 * bs / gen["mean_ms"], 3),
                        "tokens_per_s": round(
                            1000 * bs * gen["new_tokens"] / gen["mean_ms"], 3
                        ),
                    },
                }
            )
        return rows

    records = []

    for h, w in image_sizes:
        print(f"\n--- image size {h}×{w} ---")
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

        # Speedup is computed relative to baseline within each image-size group.
        baseline_map = {
            r["batch_size"]: r for r in size_records if r["backend"] == "baseline"
        }
        for r in size_records:
            ref = baseline_map[r["batch_size"]]
            r["speedup_vs_baseline"] = {
                "encoder": round(
                    ref["encoder"]["mean_ms"] / r["encoder"]["mean_ms"], 3
                ),
                "generate": round(
                    ref["generate"]["mean_ms"] / r["generate"]["mean_ms"], 3
                ),
            }

        records.extend(size_records)

    save_json(
        args.out / "bench_speed.json",
        {"meta": run_meta(args, model_id), "records": records},
    )

    print(
        f"\n{'size':>12} {'backend':>10} {'bs':>3} {'enc ms':>9} {'enc σ':>7} "
        f"{'gen ms':>9} {'gen σ':>7} {'img/s':>7} {'enc x':>6} {'gen x':>6}"
    )
    for r in records:
        size_str = f"{r['image_height']}×{r['image_width']}"
        print(
            f"{size_str:>12} {r['backend']:>10} {r['batch_size']:>3} "
            f"{r['encoder']['mean_ms']:>9.2f} {r['encoder']['std_ms']:>7.2f} "
            f"{r['generate']['mean_ms']:>9.2f} {r['generate']['std_ms']:>7.2f} "
            f"{r['throughput']['images_per_s']:>7.1f} "
            f"{r['speedup_vs_baseline']['encoder']:>6.2f} "
            f"{r['speedup_vs_baseline']['generate']:>6.2f}"
        )


if __name__ == "__main__":
    main()
