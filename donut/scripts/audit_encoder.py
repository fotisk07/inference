"""Quantify encoder output divergence: eager vs SDPA-patched, synthetic data.

Runs the same synthetic pixel_values through the eager and SDPA-patched encoder
on one model instance and reports per-sample error statistics.

Outputs:
    results/audit_encoder.json  per-sample + aggregate diff stats, run metadata
    results/audit_encoder.npz   |diff| histogram (log-spaced), per-token max
                                error map, first sample's hidden states

Usage:
    uv run python scripts/audit_encoder.py                # real model, auto device
    uv run python scripts/audit_encoder.py --tiny         # offline smoke run
"""

import numpy as np
import torch
from _common import base_parser, load_baseline_model, run_meta, save_json

from donut.accel import apply_accel
from donut.audit import diff_stats, eager_encoder
from donut.synthetic import make_pixel_values


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--n-samples", type=int, default=8)
    args = parser.parse_args()

    model, model_id = load_baseline_model(args)
    apply_accel(model, "sdpa")

    per_sample = []
    err_chunks = []
    per_token_max = []
    eager_sample = sdpa_sample = None
    for i in range(args.n_samples):
        seed = args.seed + i
        pixel_values = make_pixel_values(model, batch_size=args.batch_size, seed=seed)
        with torch.no_grad():
            with eager_encoder(model):
                ref = model.encoder(pixel_values, return_dict=True).last_hidden_state
            acc = model.encoder(pixel_values, return_dict=True).last_hidden_state
        per_sample.append({"seed": seed, **diff_stats(ref, acc)})
        err = (ref.float() - acc.float()).abs().cpu()
        err_chunks.append(err.flatten())
        per_token_max.append(err.amax(dim=-1)[0])  # (seq_len,) for batch element 0
        if eager_sample is None:
            eager_sample = ref[0].float().cpu().numpy()
            sdpa_sample = acc[0].float().cpu().numpy()

    keys = [k for k in per_sample[0] if k != "seed"]
    aggregate = {
        "mean": {k: float(np.mean([s[k] for s in per_sample])) for k in keys},
        "worst": {k: float(np.max([s[k] for s in per_sample])) for k in keys},
    }

    err_all = torch.cat(err_chunks).numpy()
    nonzero = err_all[err_all > 0]
    lo = float(nonzero.min()) if nonzero.size else 1e-12
    hi = max(float(err_all.max()), lo * 10)
    edges = np.logspace(np.log10(lo), np.log10(hi), 201)
    counts, _ = np.histogram(err_all, bins=edges)

    args.out.mkdir(parents=True, exist_ok=True)
    save_json(
        args.out / "audit_encoder.json",
        {
            "meta": run_meta(args, model_id),
            "per_sample": per_sample,
            "aggregate": aggregate,
        },
    )
    np.savez(
        args.out / "audit_encoder.npz",
        err_hist_counts=counts,
        err_hist_edges=edges,
        per_token_max_ae=torch.stack(per_token_max).numpy(),
        eager_sample=eager_sample,
        sdpa_sample=sdpa_sample,
    )
    print(f"wrote {args.out / 'audit_encoder.npz'}")

    print(f"\nEncoder eager vs SDPA over {args.n_samples} synthetic samples:")
    for k in keys:
        print(
            f"  {k:>10}: mean={aggregate['mean'][k]:.3e}  worst={aggregate['worst'][k]:.3e}"
        )


if __name__ == "__main__":
    main()
