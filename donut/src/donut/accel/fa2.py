"""Flash Attention 2 for the MBart decoder.

Only the decoder benefits: DonutSwinSelfAttention is a legacy class with no
FA2 dispatch interface. The encoder is patched with SDPA instead (see sdpa.py).

FA2 is ideal for MBart's cross-attention at TPOT steps: Q_len=1, K_len=4800,
head_dim=64, bfloat16 — all supported by flash-attn 2.x.
"""

from donut.accel.registry import register
from donut.accel.sdpa import _DecoderAttnImpl


def activate_decoder_fa2(model) -> None:
    """Activate Flash Attention 2 for the MBart decoder.

    Encoder is automatically patched with SDPA by apply_accel (fa2 backend).
    Model must already be in bfloat16 or float16 — FA2 does not support float32.
    """
    model.decoder.config._attn_implementation = "flash_attention_2"


@register
class DecoderFA2(_DecoderAttnImpl):
    """Activate Flash Attention 2 on the MBart decoder (cross-attention TPOT).

    Pair with EncoderSDPA — the Swin encoder has no FA2 dispatch path. Requires
    bfloat16/float16 weights.
    """

    name = "decoder_fa2"
    impl = "flash_attention_2"
