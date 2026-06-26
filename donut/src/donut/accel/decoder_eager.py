"""Force the MBart decoder onto the eager attention path.

transformers v5 defaults a freshly built/loaded MBart decoder to
_attn_implementation="sdpa". The "baseline" and "eager" presets must therefore
*explicitly* select eager (and assert it) rather than assuming the load-time
default — otherwise they silently run SDPA. Same flip-the-config-flag mechanism
as decoder_sdpa.py; MBart reads config._attn_implementation per call.
"""


def apply_decoder_eager(model) -> None:
    """Activate eager attention on the MBart decoder. Idempotent."""
    if not hasattr(model.decoder, "_prev_attn_impl"):
        model.decoder._prev_attn_impl = model.decoder.config._attn_implementation
    model.decoder.config._attn_implementation = "eager"


def revert_decoder_eager(model) -> None:
    """Restore the attention implementation active before apply. No-op if not applied."""
    if not hasattr(model.decoder, "_prev_attn_impl"):
        return
    model.decoder.config._attn_implementation = model.decoder._prev_attn_impl
    del model.decoder._prev_attn_impl


def check_decoder_eager(model) -> None:
    """Assert the decoder dispatches to the eager path."""
    impl = model.decoder.config._attn_implementation
    assert impl == "eager", f"Decoder attn_implementation is {impl!r}, expected 'eager'"
