"""Structural audit: apply -> check -> revert round-trips for every preset."""

import pytest
import torch

from donut.accel import apply_accel, check_accel, revert_accel

BACKENDS = ["eager", "sdpa"]  # "fa" needs CUDA + flash-attn; covered on GPU box


@torch.no_grad()
def _snapshot(model, pixel_values):
    enc = model.encoder(pixel_values, return_dict=True).last_hidden_state
    seq = model.generate(
        pixel_values=pixel_values, max_new_tokens=8, do_sample=False, use_cache=True
    )
    return enc, seq


def _assert_clean(model, impl_before):
    assert model._accel_reverts == []
    assert model.decoder.config._attn_implementation == impl_before
    assert not hasattr(model.decoder, "_prev_attn_impl")
    for stage in model.encoder.encoder.layers:
        for block in stage.blocks:
            assert not hasattr(block, "_mask_cache_applied")
            assert "get_attn_mask" not in block.__dict__
            assert not hasattr(block.attention.self, "_sdpa_patched")


@pytest.mark.parametrize("backend", BACKENDS)
def test_apply_then_check_passes(tiny_model, backend):
    apply_accel(tiny_model, backend)
    check_accel(tiny_model, backend)


def test_check_missing_backend_raises(tiny_model):
    apply_accel(tiny_model, "eager")
    with pytest.raises(AssertionError):
        check_accel(tiny_model, "sdpa")


@pytest.mark.parametrize("backend", BACKENDS)
def test_revert_restores_exact_outputs(tiny_model, pixel_values, backend):
    impl_before = tiny_model.decoder.config._attn_implementation
    enc_before, seq_before = _snapshot(tiny_model, pixel_values)

    apply_accel(tiny_model, backend)
    revert_accel(tiny_model)

    enc_after, seq_after = _snapshot(tiny_model, pixel_values)
    assert torch.equal(enc_before, enc_after)
    assert torch.equal(seq_before, seq_after)
    _assert_clean(tiny_model, impl_before)


def test_apply_idempotent(tiny_model, pixel_values):
    apply_accel(tiny_model, "sdpa")
    with torch.no_grad():
        out_once = tiny_model.encoder(pixel_values, return_dict=True).last_hidden_state
    apply_accel(tiny_model, "sdpa")
    check_accel(tiny_model, "sdpa")
    with torch.no_grad():
        out_twice = tiny_model.encoder(pixel_values, return_dict=True).last_hidden_state
    assert torch.equal(out_once, out_twice)
    # A single revert cleans up both applies (duplicate reverts are no-ops).
    impl_before = tiny_model.decoder.config._attn_implementation
    revert_accel(tiny_model)
    _assert_clean(tiny_model, impl_before)


def test_fa2_alias_maps_to_fa_preset():
    from donut.accel import _steps

    assert _steps("fa2", compile=False) == _steps("fa", compile=False)


def test_unknown_backend_raises(tiny_model):
    with pytest.raises(ValueError, match="Unknown backend"):
        apply_accel(tiny_model, "bogus")


def test_output_attentions_falls_back_to_eager(tiny_model, pixel_values):
    apply_accel(tiny_model, "sdpa")
    with torch.no_grad():
        out = tiny_model.encoder(pixel_values, output_attentions=True, return_dict=True)
    assert out.attentions is not None
    assert all(a is not None for a in out.attentions)
