"""Force PyTorch's SDPA dispatcher to a specific backend for one call site.

Named backends here are PyTorch's *native* SDPA kernels (torch.nn.attention.
SDPBackend) -- distinct from the "fa" preset (decoder_fa.py), which dispatches
to the separate flash-attn-4/CUTLASS kernel via transformers'
"flash_attention_4" attn_implementation, not through SDPA at all.
"""

from contextlib import nullcontext

from torch.nn.attention import SDPBackend, sdpa_kernel

_BACKENDS = {
    "math": SDPBackend.MATH,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}


def sdpa_backend(name: str | None):
    """Context manager restricting SDPA to one backend; nullcontext if name is None.

    Wrap it tightly around the call you're measuring, e.g.:
        with sdpa_backend("flash"):
            model.encoder(pixel_values)
    Works for both the encoder's custom monkeypatched forward (encoder_sdpa.py)
    and the decoder's built-in transformers dispatch (decoder_sdpa.py) -- both
    just call F.scaled_dot_product_attention under the hood, and sdpa_kernel is
    thread-local, so one helper covers either call site.

    Raises RuntimeError from PyTorch if the forced backend can't handle the
    given shapes/dtype -- that failure is itself useful diagnostic signal.
    """
    if name is None:
        return nullcontext()
    return sdpa_kernel([_BACKENDS[name]])
