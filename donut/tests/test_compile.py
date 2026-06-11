"""torch.compile smoke test. Slow (first-call compilation), deselected by default."""

import pytest
import torch

from donut.accel.compile import apply_compile, check_compile, revert_compile


@pytest.mark.slow
def test_compile_apply_generate_revert(tiny_model, pixel_values):
    kwargs = dict(pixel_values=pixel_values, max_new_tokens=4, do_sample=False)
    baseline = tiny_model.generate(**kwargs)
    encoder_orig, decoder_orig = tiny_model.encoder, tiny_model.decoder

    apply_compile(tiny_model)
    check_compile(tiny_model)
    compiled = tiny_model.generate(**kwargs)
    assert torch.equal(compiled, baseline)

    revert_compile(tiny_model)
    assert tiny_model.encoder is encoder_orig
    assert tiny_model.decoder is decoder_orig
