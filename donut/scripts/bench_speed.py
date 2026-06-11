"""Latency/throughput benchmark: baseline vs accelerated backends, synthetic data.

For each backend the preset is applied, structurally verified with check_accel
(never bench an unverified config), benchmarked, and fully reverted. A
"baseline" row (no optimizations at all, not even mask caching) is always
included as the speedup reference.

Outputs:
    results/bench_speed.json    {meta, records: [...]}, pd.json_normalize-friendly

Usage:
    uv run python scripts/bench_speed.py --backends eager,sdpa,fa --batch-sizes 1,2,4
    uv run python scripts/bench_speed.py --tiny --backends eager,sdpa --n-runs 3
"""

from _common import base_parser, load_baseline_model, run_meta, save_json

from donut.accel import apply_accel, check_accel, fa_available, revert_accel
from donut.bench import bench_encoder, bench_generate


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--backends", default="eager,sdpa,fa")
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--compile", action="store_true")
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
    if "fa" in backends and not fa_available():
        print("flash-attn unavailable (needs CUDA + flash-attn) — dropping 'fa'")
        backends = [b for b in backends if b != "fa"]

    n_runs_enc = args.n_runs or 20
    n_runs_gen = args.n_runs or 10
    n_warm_enc = args.n_warmup if args.n_warmup is not None else 3
    n_warm_gen = args.n_warmup if args.n_warmup is not None else 2

    model, model_id = load_baseline_model(args)

    def bench(backend: str, compiled: bool) -> list[dict]:
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
                n_warmup=n_warm_gen,
                n_runs=n_runs_gen,
                seed=args.seed,
            )
            rows.append(
                {
                    "backend": backend,
                    "compile": compiled,
                    "batch_size": bs,
                    "encoder": enc,
                    "generate": gen,
                    "throughput": {
                        "images_per_s": round(1000 * bs / gen["mean_ms"], 3),
                        "tokens_per_s": round(
                            1000 * bs * args.max_new_tokens / gen["mean_ms"], 3
                        ),
                    },
                }
            )
        return rows

    records = bench("baseline", compiled=False)
    for backend in backends:
        apply_accel(model, backend, compile=args.compile)
        check_accel(model, backend, compile=args.compile)
        records.extend(bench(backend, compiled=args.compile))
        revert_accel(model)

    baseline = {r["batch_size"]: r for r in records if r["backend"] == "baseline"}
    for r in records:
        ref = baseline[r["batch_size"]]
        r["speedup_vs_baseline"] = {
            "encoder": round(ref["encoder"]["mean_ms"] / r["encoder"]["mean_ms"], 3),
            "generate": round(ref["generate"]["mean_ms"] / r["generate"]["mean_ms"], 3),
        }

    save_json(
        args.out / "bench_speed.json",
        {"meta": run_meta(args, model_id), "records": records},
    )

    print(
        f"\n{'backend':>10} {'bs':>3} {'enc ms':>9} {'gen ms':>9} "
        f"{'tok/s':>8} {'enc x':>6} {'gen x':>6}"
    )
    for r in records:
        print(
            f"{r['backend']:>10} {r['batch_size']:>3} "
            f"{r['encoder']['mean_ms']:>9.2f} {r['generate']['mean_ms']:>9.2f} "
            f"{r['throughput']['tokens_per_s']:>8.1f} "
            f"{r['speedup_vs_baseline']['encoder']:>6.2f} "
            f"{r['speedup_vs_baseline']['generate']:>6.2f}"
        )


if __name__ == "__main__":
    main()
