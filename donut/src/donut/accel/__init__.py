"""Acceleration presets for the Donut model.

Each optimization is a module exposing three plain functions:

    apply_x(model)   -- apply the transform in-place (idempotent via guard attrs)
    revert_x(model)  -- undo it (no-op if not applied)
    check_x(model)   -- assert it is active (raises AssertionError with detail)

A preset is an ordered list of (apply, revert, check) steps. apply_accel runs
them in order and records the revert callables on the model so revert_accel can
undo everything in reverse order. To add a new optimization, write a module
with the three functions and add its step to PRESETS.
"""

from collections.abc import Callable

from donut.accel.decoder_eager import (
    apply_decoder_eager,
    check_decoder_eager,
    revert_decoder_eager,
)
from donut.accel.decoder_fa import (
    apply_decoder_fa,
    check_decoder_fa,
    fa_available,
    revert_decoder_fa,
)
from donut.accel.decoder_sdpa import (
    apply_decoder_sdpa,
    apply_decoder_sdpa_cudnn,
    apply_decoder_sdpa_efficient,
    apply_decoder_sdpa_flash,
    apply_decoder_sdpa_math,
    check_decoder_sdpa,
    check_decoder_sdpa_cudnn,
    check_decoder_sdpa_efficient,
    check_decoder_sdpa_flash,
    check_decoder_sdpa_math,
    revert_decoder_sdpa,
    revert_decoder_sdpa_cudnn,
    revert_decoder_sdpa_efficient,
    revert_decoder_sdpa_flash,
    revert_decoder_sdpa_math,
)
from donut.accel.encoder_sdpa import (
    apply_encoder_sdpa,
    check_encoder_sdpa,
    revert_encoder_sdpa,
)
from donut.accel.mask_cache import apply_mask_cache, check_mask_cache, revert_mask_cache
from donut.accel.sdpa_backend import sdpa_backend

Step = tuple[Callable, Callable, Callable]  # (apply, revert, check)

MASK_CACHE: Step = (apply_mask_cache, revert_mask_cache, check_mask_cache)
ENCODER_SDPA: Step = (apply_encoder_sdpa, revert_encoder_sdpa, check_encoder_sdpa)
DECODER_EAGER: Step = (apply_decoder_eager, revert_decoder_eager, check_decoder_eager)
DECODER_SDPA: Step = (apply_decoder_sdpa, revert_decoder_sdpa, check_decoder_sdpa)
DECODER_SDPA_CUDNN: Step = (
    apply_decoder_sdpa_cudnn,
    revert_decoder_sdpa_cudnn,
    check_decoder_sdpa_cudnn,
)
DECODER_SDPA_FLASH: Step = (
    apply_decoder_sdpa_flash,
    revert_decoder_sdpa_flash,
    check_decoder_sdpa_flash,
)
DECODER_SDPA_MATH: Step = (
    apply_decoder_sdpa_math,
    revert_decoder_sdpa_math,
    check_decoder_sdpa_math,
)
DECODER_SDPA_EFFICIENT: Step = (
    apply_decoder_sdpa_efficient,
    revert_decoder_sdpa_efficient,
    check_decoder_sdpa_efficient,
)
DECODER_FA: Step = (apply_decoder_fa, revert_decoder_fa, check_decoder_fa)

# "baseline"/"eager" pin the decoder to eager explicitly: transformers v5 defaults
# a fresh decoder to SDPA, so without DECODER_EAGER these presets would silently
# run SDPA (and check_accel would not catch it). The encoder needs no eager step —
# DonutSwin's attention is eager unless ENCODER_SDPA monkey-patches it.
# Mask caching follows (universally beneficial; the SDPA encoder patch consumes its
# cached bias). "sdpa" and all "sdpa_*" presets share the SAME ENCODER_SDPA step —
# DonutSwin has no flash path — so they differ ONLY in the decoder kernel.
PRESETS: dict[str, list[Step]] = {
    "baseline": [DECODER_EAGER],
    "eager": [DECODER_EAGER, MASK_CACHE],
    "sdpa": [MASK_CACHE, ENCODER_SDPA, DECODER_SDPA],
    "sdpa_cudnn": [MASK_CACHE, ENCODER_SDPA, DECODER_SDPA_CUDNN],
    "sdpa_flash": [MASK_CACHE, ENCODER_SDPA, DECODER_SDPA_FLASH],
    "sdpa_math": [MASK_CACHE, ENCODER_SDPA, DECODER_SDPA_MATH],
    "sdpa_efficient": [MASK_CACHE, ENCODER_SDPA, DECODER_SDPA_EFFICIENT],
    "fa": [MASK_CACHE, ENCODER_SDPA, DECODER_FA],
}

# RESEARCH STUB (research/compile-static-cache): once decoder_compiled is fleshed
# out, wire it like any other step and add a preset. Layers ON TOP of an attention
# backend — compile/static-cache is orthogonal to which SDPA kernel runs. Left
# commented so the bench sweep is unaffected until the stub actually compiles.
#
# from donut.accel.decoder_compiled import (
#     apply_decoder_compiled, revert_decoder_compiled, check_decoder_compiled,
# )
# DECODER_COMPILED: Step = (
#     apply_decoder_compiled, revert_decoder_compiled, check_decoder_compiled,
# )
# PRESETS["sdpa_compiled"] = [MASK_CACHE, ENCODER_SDPA, DECODER_SDPA, DECODER_COMPILED]

_ALIASES = {"fa2": "fa"}


def _steps(backend: str) -> list[Step]:
    backend = _ALIASES.get(str(backend), str(backend))
    if backend not in PRESETS:
        raise ValueError(f"Unknown backend {backend!r}; choose from {sorted(PRESETS)}")
    return list(PRESETS[backend])


def apply_accel(model, backend: str = "sdpa") -> None:
    """Apply a backend preset in-place, recording reverts on the model.

    Each apply is idempotent, so re-applying a backend (or layering presets
    that share steps) is safe. Undo everything with revert_accel.
    """
    reverts = getattr(model, "_accel_reverts", [])
    for apply_fn, revert_fn, _ in _steps(backend):
        apply_fn(model)
        reverts.append(revert_fn)
    model._accel_reverts = reverts


def revert_accel(model) -> None:
    """Revert every applied optimization, returning the model to eager state."""
    for revert_fn in reversed(getattr(model, "_accel_reverts", [])):
        revert_fn(model)
    model._accel_reverts = []


def check_accel(model, backend: str = "sdpa") -> None:
    """Assert every step of a backend preset is active. Raises AssertionError."""
    for _, _, check_fn in _steps(backend):
        check_fn(model)


def _init_legacy_baseline(model):
    model.decoder.config._attn_implementation = "eager"


def decoder_attn_impl(model) -> str:
    """The decoder's active attn_implementation — the accel subsystem owns this flag."""
    return model.decoder.config._attn_implementation


__all__ = [
    "DECODER_FA",
    "DECODER_SDPA",
    "DECODER_SDPA_CUDNN",
    "DECODER_SDPA_EFFICIENT",
    "DECODER_SDPA_FLASH",
    "DECODER_SDPA_MATH",
    "ENCODER_SDPA",
    "MASK_CACHE",
    "PRESETS",
    "Step",
    "apply_accel",
    "check_accel",
    "decoder_attn_impl",
    "fa_available",
    "revert_accel",
    "sdpa_backend",
]
