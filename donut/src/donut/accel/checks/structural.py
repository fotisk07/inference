"""Structural checks: verify that accelerations are active on a loaded model.

All functions inspect model state only — no forward pass, no data required.
Each raises AssertionError with a descriptive message on failure.

The canonical check is :func:`check_applied`, which iterates the optimizations
recorded on the model by ``apply_accel`` and runs each one's ``check_structural``.
:func:`check_backend` remains for callers that want to verify against a named
preset, and the per-feature helpers are kept for back-compat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import VisionEncoderDecoderModel

    from donut.accel import Backend


def check_applied(model: VisionEncoderDecoderModel) -> None:
    """Run ``check_structural`` for every optimization applied to the model."""
    from donut.accel.registry import check_optimizations

    check_optimizations(model)


def check_mask_cache(model: VisionEncoderDecoderModel) -> None:
    """Assert that mask caching has been applied to every shifted Swin block."""
    from donut.accel.mask_cache import MaskCache

    MaskCache().check_structural(model)


def check_sdpa(model: VisionEncoderDecoderModel) -> None:
    """Assert that SDPA is active on encoder (monkey-patched) and decoder (config)."""
    from donut.accel.sdpa import DecoderSDPA, EncoderSDPA

    EncoderSDPA().check_structural(model)
    DecoderSDPA().check_structural(model)


def check_fa2(model: VisionEncoderDecoderModel) -> None:
    """Assert that FA2 is active on decoder and SDPA on encoder."""
    from donut.accel.fa2 import DecoderFA2
    from donut.accel.sdpa import EncoderSDPA

    EncoderSDPA().check_structural(model)
    DecoderFA2().check_structural(model)


def check_compiled(model: VisionEncoderDecoderModel) -> None:
    """Assert that both encoder and decoder have been wrapped with torch.compile."""
    from donut.accel.compile import Compile

    Compile().check_structural(model)


def check_backend(model: VisionEncoderDecoderModel, backend: "Backend") -> None:
    """Assert that the model state matches the given backend preset.

    EAGER: only mask caching applied.
    SDPA:  mask caching + SDPA on encoder and decoder.
    FA2:   mask caching + SDPA on encoder + FA2 on decoder.
    """
    from donut.accel import Backend, preset

    backend = Backend(backend)
    for opt in preset(backend):
        opt.check_structural(model)
