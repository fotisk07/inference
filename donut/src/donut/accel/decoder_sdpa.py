"""SDPA for the MBart decoder via its built-in attention dispatch.

Unlike the Swin encoder (encoder_sdpa.py), MBart reads
config._attn_implementation per call, so flipping the config flag is enough.
Only ever assign on model.decoder.config — in transformers v5 the top-level
config setter recursively touches sub-configs.
"""

from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward


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


def _make_sdpa_variant(
    name: str, backends: list[SDPBackend], *, set_priority: bool = False
):
    """Register attn_implementation f"sdpa_{name}", restricted to `backends`.

    Registers a new named attn_implementation, scoped to this one wrapper
    function -- does not touch the existing "sdpa" entry, so it can't affect
    the encoder (which never reads this registry; DonutSwin is patched
    directly in encoder_sdpa.py) or any other model in the process using
    plain "sdpa". Returns (apply, revert, check) functions sharing the same
    _prev_attn_impl save/restore dance as apply_decoder_sdpa.
    """
    impl_name = f"sdpa_{name}"

    def _attention_forward(module, query, key, value, attention_mask, **kwargs):
        with sdpa_kernel(backends, set_priority=set_priority):
            return sdpa_attention_forward(
                module, query, key, value, attention_mask, **kwargs
            )

    AttentionInterface.register(impl_name, _attention_forward)

    def apply(model) -> None:
        if not hasattr(model.decoder, "_prev_attn_impl"):
            model.decoder._prev_attn_impl = getattr(
                model.decoder.config, "_attn_implementation", "eager"
            )
        model.decoder.config._attn_implementation = impl_name

    def revert(model) -> None:
        if not hasattr(model.decoder, "_prev_attn_impl"):
            return
        model.decoder.config._attn_implementation = model.decoder._prev_attn_impl
        del model.decoder._prev_attn_impl

    def check(model) -> None:
        impl = model.decoder.config._attn_implementation
        assert impl == impl_name, (
            f"Decoder attn_implementation is {impl!r}, expected {impl_name!r}"
        )

    apply.__name__ = f"apply_decoder_{impl_name}"
    revert.__name__ = f"revert_decoder_{impl_name}"
    check.__name__ = f"check_decoder_{impl_name}"
    return apply, revert, check


# cuDNN's SDPA backend can't handle kv_len=1 (RuntimeError) -- and generate()'s
# first decode step always has kv_len=1 before the cache grows. set_priority=True
# means: prefer cuDNN whenever it can handle the shape, transparently fall back to
# the efficient backend otherwise (it handled every shape in the H100 sweep).
apply_decoder_sdpa_cudnn, revert_decoder_sdpa_cudnn, check_decoder_sdpa_cudnn = (
    _make_sdpa_variant(
        "cudnn",
        [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],
        set_priority=True,
    )
)

# flash/math/efficient need no fallback: the H100 kernel sweep
# (scripts/bench_attention_kernels.py) showed all three return a real number at
# every (mode, kv_len, batch_size) row tested, including kv_len=1 decode.
apply_decoder_sdpa_flash, revert_decoder_sdpa_flash, check_decoder_sdpa_flash = (
    _make_sdpa_variant("flash", [SDPBackend.FLASH_ATTENTION])
)
apply_decoder_sdpa_math, revert_decoder_sdpa_math, check_decoder_sdpa_math = (
    _make_sdpa_variant("math", [SDPBackend.MATH])
)
(
    apply_decoder_sdpa_efficient,
    revert_decoder_sdpa_efficient,
    check_decoder_sdpa_efficient,
) = _make_sdpa_variant("efficient", [SDPBackend.EFFICIENT_ATTENTION])
