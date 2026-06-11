"""SDPA for the MBart decoder via its built-in attention dispatch.

Unlike the Swin encoder (encoder_sdpa.py), MBart reads
config._attn_implementation per call, so flipping the config flag is enough.
Only ever assign on model.decoder.config — in transformers v5 the top-level
config setter recursively touches sub-configs.
"""


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
