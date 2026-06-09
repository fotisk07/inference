from enum import Enum

from donut.accel.fa2 import activate_decoder_fa2
from donut.accel.mask_cache import apply_mask_cache
from donut.accel.sdpa import activate_decoder_sdpa, patch_swin_sdpa


class Backend(str, Enum):
    EAGER = "eager"
    SDPA = "sdpa"
    FA2 = "fa2"


def apply_accel(model, backend: "Backend | str", *, compile: bool = False) -> None:
    """Apply mask caching and the chosen attention backend in-place.

    Mask caching is always applied first — it is universally beneficial and
    required by the SDPA encoder patch. The attention backend (SDPA/FA2) is
    layered on top. Pass compile=True to additionally wrap encoder and decoder
    with torch.compile (dynamic=True).
    """
    from donut.accel.compile import apply_compile

    backend = Backend(backend)
    apply_mask_cache(model)

    if backend is Backend.EAGER:
        pass
    elif backend is Backend.SDPA:
        patch_swin_sdpa(model)
        activate_decoder_sdpa(model)
    elif backend is Backend.FA2:
        patch_swin_sdpa(model)
        activate_decoder_fa2(model)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    if compile:
        apply_compile(model)
