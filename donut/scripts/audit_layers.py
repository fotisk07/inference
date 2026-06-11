"""Localize encoder divergence: per-stage and per-block eager vs SDPA diff.

Captures every Swin stage and block output under the eager and the
SDPA-patched encoder for the same synthetic input, then reports where the
divergence first appears and how it grows through the network.

Outputs:
    results/audit_layers.csv    one row per stage/block with diff stats
    results/audit_layers.json   same rows + run metadata + first divergence locus

Usage:
    uv run python scripts/audit_layers.py
    uv run python scripts/audit_layers.py --tiny
"""

from _common import base_parser, load_baseline_model, run_meta, save_csv, save_json

from donut.accel import apply_accel
from donut.audit import capture_encoder_outputs, diff_stats, eager_encoder
from donut.synthetic import make_pixel_values


def main() -> None:
    parser = base_parser(__doc__)
    args = parser.parse_args()

    model, model_id = load_baseline_model(args)
    apply_accel(model, "sdpa")
    pixel_values = make_pixel_values(model, batch_size=args.batch_size, seed=args.seed)

    with eager_encoder(model):
        captured_eager = capture_encoder_outputs(model, pixel_values)
    captured_sdpa = capture_encoder_outputs(model, pixel_values)

    rows = []
    for ref, acc in zip(captured_eager, captured_sdpa):
        key = ("level", "stage", "block")
        assert all(ref[k] == acc[k] for k in key), "hook capture order mismatch"
        rows.append(
            {
                "level": ref["level"],
                "stage": ref["stage"],
                "block": ref["block"],
                "shift_size": ref["shift_size"],
                "num_heads": ref["num_heads"],
                "seq_len": ref["output"].shape[1],
                **diff_stats(ref["output"], acc["output"]),
            }
        )

    block_rows = [r for r in rows if r["level"] == "block"]
    first = next((r for r in block_rows if r["mean_ae"] > 0), None)
    first_locus = (
        {
            "stage": first["stage"],
            "block": first["block"],
            "shift_size": first["shift_size"],
        }
        if first
        else None
    )

    save_csv(args.out / "audit_layers.csv", rows)
    save_json(
        args.out / "audit_layers.json",
        {
            "meta": run_meta(args, model_id),
            "first_divergence": first_locus,
            "rows": rows,
        },
    )

    print("\nPer-block encoder divergence (eager vs SDPA):")
    print(f"{'loc':>12} {'shift':>5} {'mean_ae':>10} {'max_ae':>10} {'cos_sim':>9}")
    for r in block_rows:
        loc = f"s{r['stage']}b{r['block']}"
        print(
            f"{loc:>12} {r['shift_size']:>5} {r['mean_ae']:>10.3e} "
            f"{r['max_ae']:>10.3e} {r['cosine_sim']:>9.6f}"
        )
    print(f"first divergence: {first_locus}")


if __name__ == "__main__":
    main()
