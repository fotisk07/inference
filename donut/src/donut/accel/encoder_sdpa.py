"""SDPA acceleration for the DonutSwin encoder.

Monkey-patches DonutSwinSelfAttention.forward to use
F.scaled_dot_product_attention. DonutSwin is a legacy class with no attention-
dispatch interface, so attn_implementation="sdpa" at from_pretrained time has
no effect — this patch is required.

Orthogonal to the mask cache (mask_cache.py): the cyclic-shift additive bias it
produces is passed directly to attn_mask= in SDPA.
"""

import types

import torch
import torch.nn.functional as F


def _sdpa_self_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.FloatTensor | None = None,
    output_attentions: bool | None = False,
) -> tuple[torch.Tensor]:
    if output_attentions:
        # SDPA doesn't return attention weights; fall back to original eager path.
        return self._original_forward(hidden_states, attention_mask, output_attentions)

    batch_size, seq_len, _ = hidden_states.shape
    hidden_shape = (batch_size, seq_len, -1, self.attention_head_size)

    query_layer = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
    key_layer = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
    value_layer = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
    # all: (batch_size, num_heads, seq_len, head_dim)

    # Relative position bias: (1, num_heads, seq_len, seq_len) — broadcast over batch.
    rel_bias = (
        self.relative_position_bias_table[self.relative_position_index.view(-1)]
        .view(seq_len, seq_len, -1)
        .permute(2, 0, 1)
        .contiguous()
        .unsqueeze(0)
    )
    attn_mask = rel_bias

    if attention_mask is not None:
        # attention_mask from mask_cache.py: (nw, seq_len, seq_len), additive float bias.
        # Expand to (batch_size, num_heads, seq_len, seq_len).
        nw = attention_mask.shape[0]
        B = batch_size // nw
        am = (
            attention_mask.unsqueeze(1)
            .expand(nw, self.num_attention_heads, seq_len, seq_len)
            .repeat(B, 1, 1, 1)
        )
        attn_mask = rel_bias + am

    # SDPA dispatches to Flash/Efficient/Math backends based on shape and mask.
    # ws²=100 is not divisible by 8, so Flash SDP is typically not selected.
    # Efficient SDP (xformers) or Math SDP handle the float attn_mask correctly.
    # Rare: one extreme bfloat16 value may differ from the eager path by ~10 units
    # due to float32 accumulation in some backends. Decoder outputs are unaffected.
    context_layer = F.scaled_dot_product_attention(
        query_layer,
        key_layer,
        value_layer,
        attn_mask=attn_mask,
        dropout_p=self.dropout.p if self.training else 0.0,
        is_causal=False,
    )
    # (batch_size, num_heads, seq_len, head_dim)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    context_layer = context_layer.view(batch_size, seq_len, self.all_head_size)

    return (context_layer,)


def apply_encoder_sdpa(model) -> None:
    """Replace DonutSwinSelfAttention.forward with an SDPA-based implementation.

    Idempotent — a guard prevents double-patching. Must be applied after the
    mask cache, since SDPA consumes the cyclic-shift bias it produces.
    """
    for stage in model.encoder.encoder.layers:
        for block in stage.blocks:
            self_attn = block.attention.self
            if hasattr(self_attn, "_sdpa_patched"):
                continue
            self_attn._original_forward = self_attn.forward
            self_attn.forward = types.MethodType(_sdpa_self_forward, self_attn)
            self_attn._sdpa_patched = True


def revert_encoder_sdpa(model) -> None:
    """Undo apply_encoder_sdpa: restore the original eager forward on every block.

    No-op if not applied.
    """
    for stage in model.encoder.encoder.layers:
        for block in stage.blocks:
            self_attn = block.attention.self
            if not hasattr(self_attn, "_sdpa_patched"):
                continue
            # Deleting the instance forward restores the class method; the saved
            # _original_forward was that same bound method.
            del self_attn.forward
            del self_attn._original_forward
            del self_attn._sdpa_patched


def check_encoder_sdpa(model) -> None:
    """Assert the SDPA patch is active on every encoder self-attention block."""
    for i, stage in enumerate(model.encoder.encoder.layers):
        for j, block in enumerate(stage.blocks):
            self_attn = block.attention.self
            assert getattr(self_attn, "_sdpa_patched", False), (
                f"Stage {i} block {j} encoder self-attention is not SDPA-patched"
            )
