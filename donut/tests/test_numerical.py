"""Numerical audit: each optimization vs the eager baseline on the tiny model.

Float32 on CPU, so tolerances are tight. The loose bfloat16/CUDA tolerances
belong to the audit scripts (scripts/audit_*.py), not here.
"""

import torch

from donut.accel import apply_accel
from donut.accel.mask_cache import apply_mask_cache
from donut.audit import eager_encoder


@torch.no_grad()
def _encode(model, pixel_values):
    return model.encoder(pixel_values, return_dict=True).last_hidden_state


def test_mask_cache_bit_exact(tiny_model, pixel_values):
    """Caching only avoids recomputation — outputs must be bit-identical."""
    before = _encode(tiny_model, pixel_values)
    apply_mask_cache(tiny_model)
    after = _encode(tiny_model, pixel_values)
    assert torch.equal(before, after)


def test_cached_mask_values_match_reference(tiny_model):
    """The cached mask must equal the class-method computation exactly."""
    apply_mask_cache(tiny_model)
    block = next(
        block
        for stage in tiny_model.encoder.encoder.layers
        for block in stage.blocks
        if block.shift_size > 0
    )
    height, width = block.input_resolution
    device = torch.device("cpu")
    cached = block.get_attn_mask(height, width, torch.float32, device)
    reference = type(block).get_attn_mask(block, height, width, torch.float32, device)
    assert torch.equal(cached, reference)
    # Second call must hit the cache (same object back).
    assert block.get_attn_mask(height, width, torch.float32, device) is cached


def test_encoder_sdpa_close_to_eager(tiny_model, pixel_values):
    apply_accel(tiny_model, "sdpa")
    with eager_encoder(tiny_model):
        ref = _encode(tiny_model, pixel_values)
    accel = _encode(tiny_model, pixel_values)
    torch.testing.assert_close(accel, ref, atol=1e-5, rtol=1e-5)


def test_decoder_sdpa_tokens_match_eager(tiny_model, pixel_values):
    def generate(impl):
        tiny_model.decoder.config._attn_implementation = impl
        return tiny_model.generate(
            pixel_values=pixel_values,
            max_new_tokens=10,
            do_sample=False,
            use_cache=True,
        )

    assert torch.equal(generate("eager"), generate("sdpa"))
