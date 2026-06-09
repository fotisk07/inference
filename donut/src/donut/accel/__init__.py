from enum import Enum

from donut.accel.compile import Compile
from donut.accel.fa2 import DecoderFA2, activate_decoder_fa2
from donut.accel.mask_cache import MaskCache, apply_mask_cache
from donut.accel.registry import (
    REGISTRY,
    Optimization,
    applied_optimizations,
    apply_optimizations,
    check_optimizations,
    register,
    revert_optimizations,
)
from donut.accel.sdpa import (
    DecoderSDPA,
    EncoderSDPA,
    activate_decoder_sdpa,
    patch_swin_sdpa,
)


class Backend(str, Enum):
    EAGER = "eager"
    SDPA = "sdpa"
    FA2 = "fa2"


def preset(backend: "Backend | str") -> list[Optimization]:
    """Return the ordered optimization list for a backend preset.

    MaskCache is always first (universally beneficial; required by EncoderSDPA).
    The attention backend is layered on top.
    """
    backend = Backend(backend)
    if backend is Backend.EAGER:
        return [MaskCache()]
    if backend is Backend.SDPA:
        return [MaskCache(), EncoderSDPA(), DecoderSDPA()]
    if backend is Backend.FA2:
        return [MaskCache(), EncoderSDPA(), DecoderFA2()]
    raise ValueError(f"Unknown backend: {backend!r}")


def apply_accel(
    model,
    backend: "Backend | str",
    *,
    compile: bool = False,
    optimizations: "list[Optimization] | None" = None,
) -> None:
    """Apply an ordered list of optimizations in-place and record them on the model.

    By default the list comes from the ``backend`` preset (EAGER/SDPA/FA2). Pass
    ``optimizations`` to supply an explicit composed list instead (``backend`` is
    then ignored for selection). ``compile=True`` appends a Compile step.

    Applied optimizations are tracked so they can be reverted with
    :func:`revert_accel` and verified with the structural checks — no need to
    order backends least->most aggressive or reload the model between configs.
    """
    opts = list(optimizations) if optimizations is not None else preset(backend)
    if compile:
        opts.append(Compile())
    apply_optimizations(model, opts)


def revert_accel(model) -> None:
    """Revert every applied optimization, returning the model to eager state."""
    revert_optimizations(model)


__all__ = [
    "Backend",
    "Optimization",
    "REGISTRY",
    "register",
    "preset",
    "apply_accel",
    "revert_accel",
    "applied_optimizations",
    "check_optimizations",
    "MaskCache",
    "EncoderSDPA",
    "DecoderSDPA",
    "DecoderFA2",
    "Compile",
    # legacy function API (kept for back-compat)
    "apply_mask_cache",
    "patch_swin_sdpa",
    "activate_decoder_sdpa",
    "activate_decoder_fa2",
]
