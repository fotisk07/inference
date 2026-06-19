"""SDPA for the MBart decoder via its built-in attention dispatch.

Unlike the Swin encoder (encoder_sdpa.py), MBart reads
config._attn_implementation per call, so flipping the config flag is enough.
Only ever assign on model.decoder.config — in transformers v5 the top-level
config setter recursively touches sub-configs.
"""

from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from donut.accel.sdpa_backend import sdpa_backend


def apply_decoder_sdpa(model) -> None:
    """Activate PyTorch SDPA on the MBart decoder. Idempotent."""
    if not hasattr(model.decoder, "_prev_attn_impl"):
        model.decoder._prev_attn_impl = getattr(
            model.decoder.config, "_attn_implementation", "eager"
        )
    model.decoder.config._attn_implementation = "sdpa"


def revert_decoder_sdpa(model) -> None:
    """Restore the attention implementation active before apply. No-op if not applied."""
    if not hasattr(model.decoder, "_prev_attn_impl"):
        return
    model.decoder.config._attn_implementation = model.decoder._prev_attn_impl
    del model.decoder._prev_attn_impl


def check_decoder_sdpa(model) -> None:
    """Assert the decoder dispatches to SDPA."""
    impl = model.decoder.config._attn_implementation
    assert impl == "sdpa", f"Decoder attn_implementation is {impl!r}, expected 'sdpa'"


def _sdpa_cudnn_attention_forward(module, query, key, value, attention_mask, **kwargs):
    with sdpa_backend("cudnn"):
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )


# Registers a new named attn_implementation, scoped to this one wrapper function --
# does not touch the existing "sdpa" entry, so it can't affect the encoder (which
# never reads this registry; DonutSwin is patched directly in encoder_sdpa.py) or
# any other model in the process using plain "sdpa".
AttentionInterface.register("sdpa_cudnn", _sdpa_cudnn_attention_forward)


def apply_decoder_sdpa_cudnn(model) -> None:
    """Activate SDPA restricted to the cuDNN backend on the MBart decoder. Idempotent."""
    if not hasattr(model.decoder, "_prev_attn_impl"):
        model.decoder._prev_attn_impl = getattr(
            model.decoder.config, "_attn_implementation", "eager"
        )
    model.decoder.config._attn_implementation = "sdpa_cudnn"


def revert_decoder_sdpa_cudnn(model) -> None:
    """Restore the attention implementation active before apply. No-op if not applied."""
    if not hasattr(model.decoder, "_prev_attn_impl"):
        return
    model.decoder.config._attn_implementation = model.decoder._prev_attn_impl
    del model.decoder._prev_attn_impl


def check_decoder_sdpa_cudnn(model) -> None:
    """Assert the decoder dispatches to the cuDNN-restricted SDPA path."""
    impl = model.decoder.config._attn_implementation
    assert impl == "sdpa_cudnn", (
        f"Decoder attn_implementation is {impl!r}, expected 'sdpa_cudnn'"
    )
