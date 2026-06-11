"""Flash attention for the MBart decoder.

Only the decoder benefits: DonutSwin has no flash-attention dispatch interface;
the encoder is patched with SDPA instead (encoder_sdpa.py). Flash attention is
ideal for MBart's cross-attention at TPOT steps: Q_len=1, K_len=4800,
head_dim=64, bfloat16.

The impl string is chosen from what is actually installed: transformers
dispatches "flash_attention_2" to the flash-attn 2.x package and
"flash_attention_4" to flash-attn-4 — they are not interchangeable.
"""

import torch
from transformers.utils import is_flash_attn_2_available, is_flash_attn_4_available

_FA_IMPLS = ("flash_attention_2", "flash_attention_4")


def _fa_impl() -> str | None:
    """Return the attn_implementation string for the installed flash-attn, or None."""
    if is_flash_attn_2_available():
        return "flash_attention_2"
    if is_flash_attn_4_available():
        return "flash_attention_4"
    return None


def fa_available() -> bool:
    """True when CUDA and a flash-attn package transformers can dispatch to exist."""
    return torch.cuda.is_available() and _fa_impl() is not None


def apply_decoder_fa(model) -> None:
    """Activate flash attention on the MBart decoder. Idempotent.

    Raises RuntimeError when flash attention is unavailable — callers decide
    whether to skip (no silent fallback). Model must be bfloat16/float16.
    """
    if not fa_available():
        raise RuntimeError(
            "Flash attention unavailable: requires CUDA and the flash-attn package "
            "(donut[fa2] extra). Use backend='sdpa' instead."
        )
    if not hasattr(model.decoder, "_prev_attn_impl"):
        model.decoder._prev_attn_impl = getattr(
            model.decoder.config, "_attn_implementation", "eager"
        )
    model.decoder.config._attn_implementation = _fa_impl()


def revert_decoder_fa(model) -> None:
    """Restore the attention implementation active before apply. No-op if not applied."""
    if not hasattr(model.decoder, "_prev_attn_impl"):
        return
    model.decoder.config._attn_implementation = model.decoder._prev_attn_impl
    del model.decoder._prev_attn_impl


def check_decoder_fa(model) -> None:
    """Assert the decoder dispatches to a flash-attention implementation."""
    impl = model.decoder.config._attn_implementation
    assert impl in _FA_IMPLS, (
        f"Decoder attn_implementation is {impl!r}, expected one of {_FA_IMPLS}"
    )
