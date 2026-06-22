"""Audit helpers: quantify divergence between baseline and accelerated paths.

All comparisons are computed in float32 regardless of model dtype. Used by
scripts/audit_accel.py and the test suite.
"""

from contextlib import contextmanager

import torch
import torch.nn.functional as F


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Error statistics between two same-shaped tensors (a = baseline)."""
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    err = (a - b).abs()
    n = err.numel()
    return {
        "max_ae": err.max().item(),
        "mean_ae": err.mean().item(),
        "p50_ae": err.median().item(),
        "p99_ae": err.kthvalue(max(1, int(n * 0.99))).values.item(),
        "cosine_sim": F.cosine_similarity(a, b, dim=0).item(),
        "rel_fro": (
            torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a)
        ).item(),
    }


@contextmanager
def eager_encoder(model):
    """Temporarily restore the original eager forward on SDPA-patched blocks.

    If the encoder SDPA patch is applied, each block stores _original_forward.
    This context manager swaps it back for the duration, allowing a fair eager
    vs. accelerated comparison without loading a second model.
    """
    swapped = []
    for stage in model.encoder.encoder.layers:
        for block in stage.blocks:
            sa = block.attention.self
            if hasattr(sa, "_sdpa_patched"):
                swapped.append((sa, sa.forward))
                sa.forward = sa._original_forward
    try:
        yield
    finally:
        for sa, fwd in swapped:
            sa.forward = fwd
