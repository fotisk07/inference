"""Quantify decoder logits divergence and its accumulation across decoding steps.

Two modes (--encoder):
    eager  both sides get the SAME eager encoder output — isolates the
           decoder-intrinsic eager vs SDPA divergence.
    sdpa   baseline = eager encoder + eager decoder, accelerated = SDPA encoder
           + SDPA decoder — measures how encoder divergence propagates and
           compounds over decoding steps (end-to-end).

The baseline side drives the greedy token trajectory; both sides decode with
KV cache. Per step: logits diff, KL, top-1/top-5 agreement, and the baseline
top1-top2 margin (how close divergence came to flipping a token).

Outputs:
    results/audit_decoder_<mode>.csv    one row per decoding step
    results/audit_decoder_<mode>.json   summary + run metadata

Usage:
    uv run python scripts/audit_decoder.py --encoder eager
    uv run python scripts/audit_decoder.py --encoder sdpa --n-steps 128
    uv run python scripts/audit_decoder.py --tiny --encoder eager
"""

import torch
from _common import base_parser, load_baseline_model, run_meta, save_csv, save_json

from donut.accel import apply_accel
from donut.audit import eager_encoder, stepwise_decode_compare
from donut.synthetic import make_pixel_values


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--encoder", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--n-steps", type=int, default=64)
    args = parser.parse_args()

    model, model_id = load_baseline_model(args)
    apply_accel(model, "sdpa")
    # Stepwise comparison is single-sequence; --batch-size is ignored here.
    pixel_values = make_pixel_values(model, batch_size=1, seed=args.seed)

    with torch.no_grad():
        with eager_encoder(model):
            enc_eager = model.encoder(pixel_values, return_dict=True).last_hidden_state
        enc_sdpa = model.encoder(pixel_values, return_dict=True).last_hidden_state
    enc_accel = enc_eager if args.encoder == "eager" else enc_sdpa

    rows = stepwise_decode_compare(
        model,
        enc_eager,
        enc_accel,
        impl_a="eager",
        impl_b="sdpa",
        n_steps=args.n_steps,
    )

    first_mismatch = next((r["step"] for r in rows if not r["top1_match"]), None)
    summary = {
        "mode": args.encoder,
        "n_steps": args.n_steps,
        "first_token_mismatch_step": first_mismatch,
        "top1_match_rate": sum(r["top1_match"] for r in rows) / len(rows),
        "max_logits_max_ae": max(r["logits_max_ae"] for r in rows),
        "final_step_kl": rows[-1]["kl_div"],
        "min_margin_a": min(r["margin_a"] for r in rows),
    }

    save_csv(args.out / f"audit_decoder_{args.encoder}.csv", rows)
    save_json(
        args.out / f"audit_decoder_{args.encoder}.json",
        {"meta": run_meta(args, model_id), "summary": summary},
    )

    print(f"\nDecoder divergence, encoder mode={args.encoder}, {args.n_steps} steps:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
