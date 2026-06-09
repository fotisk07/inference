"""Structural checks: verify that accelerations are active on a loaded model.

All functions inspect model state only — no forward pass, no data required.
Each raises AssertionError with a descriptive message on failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import VisionEncoderDecoderModel

    from donut.accel import Backend


def check_mask_cache(model: VisionEncoderDecoderModel) -> None:
    """Assert that mask caching has been applied to every shifted Swin block."""
    for i, stage in enumerate(model.encoder.encoder.layers):
        for j, block in enumerate(stage.blocks):
            if block.shift_size == 0:
                continue
            assert getattr(block, "_mask_cache_applied", False), (
                f"Stage {i} block {j} (shift_size={block.shift_size}) "
                "does not have mask caching applied — call apply_mask_cache() first"
            )


def check_sdpa(model: VisionEncoderDecoderModel) -> None:
    """Assert that SDPA is active on encoder (monkey-patched) and decoder (config)."""
    for i, stage in enumerate(model.encoder.encoder.layers):
        for j, block in enumerate(stage.blocks):
            self_attn = block.attention.self
            assert getattr(self_attn, "_sdpa_patched", False), (
                f"Stage {i} block {j} encoder self-attention is not SDPA-patched"
            )
    impl = model.decoder.config._attn_implementation
    assert impl == "sdpa", f"Decoder attn_implementation is {impl!r}, expected 'sdpa'"


def check_fa2(model: VisionEncoderDecoderModel) -> None:
    """Assert that FA2 is active on decoder and SDPA on encoder."""
    for i, stage in enumerate(model.encoder.encoder.layers):
        for j, block in enumerate(stage.blocks):
            self_attn = block.attention.self
            assert getattr(self_attn, "_sdpa_patched", False), (
                f"Stage {i} block {j} encoder self-attention is not SDPA-patched "
                "(FA2 backend requires SDPA on encoder)"
            )
    impl = model.decoder.config._attn_implementation
    assert impl == "flash_attention_2", (
        f"Decoder attn_implementation is {impl!r}, expected 'flash_attention_2'"
    )


def check_compiled(model: VisionEncoderDecoderModel) -> None:
    """Assert that both encoder and decoder have been wrapped with torch.compile."""
    assert getattr(model.encoder, "_compiled", False), (
        "Encoder has not been compiled — call apply_compile() first"
    )
    assert getattr(model.decoder, "_compiled", False), (
        "Decoder has not been compiled — call apply_compile() first"
    )


def check_backend(model: VisionEncoderDecoderModel, backend: "Backend") -> None:
    """Assert that the model state matches the given backend.

    EAGER: only mask caching applied.
    SDPA:  mask caching + SDPA on encoder and decoder.
    FA2:   mask caching + SDPA on encoder + FA2 on decoder.
    """
    from donut.accel import Backend

    backend = Backend(backend)
    check_mask_cache(model)
    if backend is Backend.EAGER:
        return
    if backend is Backend.SDPA:
        check_sdpa(model)
        return
    if backend is Backend.FA2:
        check_fa2(model)
        return
    raise ValueError(f"Unknown backend: {backend!r}")
