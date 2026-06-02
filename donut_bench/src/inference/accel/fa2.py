"""Flash Attention 2 for the MBart decoder.

Only the decoder benefits: DonutSwinSelfAttention is a legacy class with no
FA2 dispatch interface. The encoder is patched with SDPA instead (see sdpa.py).

FA2 is ideal for MBart's cross-attention at TPOT steps: Q_len=1, K_len=4800,
head_dim=64, bfloat16 — all supported by flash-attn 2.x.
"""


def activate_decoder_fa2(model) -> None:
    """Activate Flash Attention 2 for the MBart decoder.

    Encoder is automatically patched with SDPA by apply_accel (fa2 backend).
    Model must already be in bfloat16 or float16 — FA2 does not support float32.
    """
    model.decoder.config._attn_implementation = "flash_attention_2"
