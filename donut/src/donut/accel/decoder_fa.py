import torch
from transformers.utils import is_flash_attn_4_available


def fa_available() -> bool:
    """True when CUDA and a flash-attn package transformers can dispatch to exist."""
    return torch.cuda.is_available() and is_flash_attn_4_available()


def apply_decoder_fa(model) -> None:
    """Activate flash attention on the MBart decoder. Idempotent.

    Raises RuntimeError when flash attention is unavailable — callers decide
    whether to skip (no silent fallback). Model must be bfloat16/float16.
    """
    if not fa_available():
        raise RuntimeError(
            "Flash attention unavailable: requires CUDA and the flash-attn package "
        )
    if not hasattr(model.decoder, "_prev_attn_impl"):
        model.decoder._prev_attn_impl = getattr(
            model.decoder.config, "_attn_implementation", "eager"
        )
    model.decoder.config._attn_implementation = "flash_attention_4"


def revert_decoder_fa(model) -> None:
    """Restore the attention implementation active before apply. No-op if not applied."""
    if not hasattr(model.decoder, "_prev_attn_impl"):
        return
    model.decoder.config._attn_implementation = model.decoder._prev_attn_impl
    del model.decoder._prev_attn_impl


def check_decoder_fa(model) -> None:
    """Assert the decoder dispatches to a flash-attention implementation."""
    impl = model.decoder.config._attn_implementation
    assert impl == "flash_attention_4"(f"Decoder attn_implementation is {impl!r}")
