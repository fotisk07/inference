"""Audit helpers: quantify divergence between baseline and accelerated paths.

All comparisons are computed in float32 regardless of model dtype. Used by the
scripts in scripts/audit_*.py and the test suite.
"""

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from donut.synthetic import make_decoder_input_ids


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


def capture_encoder_outputs(model, pixel_values: torch.Tensor) -> list[dict]:
    """Run one encoder forward, capturing each stage and block output.

    Returns rows of {level, stage, block, shift_size, num_heads, output} where
    output is the module's hidden states (float32, CPU). Compare two captures
    (eager vs accelerated) row-by-row with diff_stats to localize divergence.
    """
    rows: list[dict] = []
    handles = []

    def add_hook(module, meta):
        def hook(_mod, _args, output):
            out = output[0] if isinstance(output, tuple) else output
            rows.append({**meta, "output": out.detach().float().cpu()})

        handles.append(module.register_forward_hook(hook))

    for i, stage in enumerate(model.encoder.encoder.layers):
        for j, block in enumerate(stage.blocks):
            add_hook(
                block,
                {
                    "level": "block",
                    "stage": i,
                    "block": j,
                    "shift_size": int(block.shift_size),
                    "num_heads": int(block.attention.self.num_attention_heads),
                },
            )
        add_hook(
            stage,
            {
                "level": "stage",
                "stage": i,
                "block": None,
                "shift_size": None,
                "num_heads": None,
            },
        )

    try:
        with torch.no_grad():
            model.encoder(pixel_values, return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return rows


@torch.no_grad()
def stepwise_decode_compare(
    model,
    enc_a: torch.Tensor,
    enc_b: torch.Tensor,
    *,
    impl_a: str = "eager",
    impl_b: str = "sdpa",
    n_steps: int = 64,
    bos_id: int | None = None,
) -> list[dict]:
    """Greedy-decode stepwise, comparing two decoder configurations per step.

    Side A (baseline: encoder hidden states enc_a, decoder attention impl_a)
    drives the token trajectory. Side B is advanced along the same prefix with
    its own KV cache, so each step compares logits for an identical context.
    Records per-step logit divergence and how close it comes to flipping the
    decoded token (margin = baseline top1 - top2 logit gap).

    Batch size must be 1. enc_a and enc_b may be the same tensor (isolates
    decoder-intrinsic divergence) or eager vs accelerated encoder outputs
    (measures end-to-end accumulation).
    """
    assert enc_a.shape[0] == 1 and enc_b.shape[0] == 1, "batch_size must be 1"
    dec = model.decoder
    saved_impl = dec.config._attn_implementation
    tok = make_decoder_input_ids(model, batch_size=1, bos_id=bos_id)
    cache_a = cache_b = None
    rows: list[dict] = []
    try:
        for step in range(n_steps):
            dec.config._attn_implementation = impl_a
            out_a = dec(
                input_ids=tok,
                encoder_hidden_states=enc_a,
                past_key_values=cache_a,
                use_cache=True,
            )
            cache_a = out_a.past_key_values
            logits_a = out_a.logits[:, -1, :].float()

            dec.config._attn_implementation = impl_b
            out_b = dec(
                input_ids=tok,
                encoder_hidden_states=enc_b,
                past_key_values=cache_b,
                use_cache=True,
            )
            cache_b = out_b.past_key_values
            logits_b = out_b.logits[:, -1, :].float()

            token_a = int(logits_a.argmax(-1))
            token_b = int(logits_b.argmax(-1))
            top2 = logits_a.topk(2, dim=-1).values[0]
            k = min(5, logits_a.shape[-1])
            top5_a = set(logits_a.topk(k, dim=-1).indices[0].tolist())
            top5_b = set(logits_b.topk(k, dim=-1).indices[0].tolist())
            err = (logits_a - logits_b).abs()
            kl = F.kl_div(
                F.log_softmax(logits_b, dim=-1),
                F.log_softmax(logits_a, dim=-1),
                log_target=True,
                reduction="batchmean",
            )
            rows.append(
                {
                    "step": step,
                    "token_a": token_a,
                    "token_b": token_b,
                    "top1_match": token_a == token_b,
                    "top5_overlap": len(top5_a & top5_b) / k,
                    "logits_max_ae": err.max().item(),
                    "logits_mean_ae": err.mean().item(),
                    "kl_div": kl.item(),
                    "margin_a": (top2[0] - top2[1]).item(),
                }
            )
            # Baseline trajectory drives both sides.
            tok = logits_a.argmax(-1, keepdim=True)
    finally:
        dec.config._attn_implementation = saved_impl
    return rows
