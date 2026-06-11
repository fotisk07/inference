"""KV-cache correctness: cached decoding must match the non-cached baseline.

All tests run on the tiny CPU model (float32), so logits comparisons can use
tight tolerances.
"""

import pytest
import torch

from donut.synthetic import make_decoder_input_ids

N_STEPS = 10
ATOL = RTOL = 1e-4


@torch.no_grad()
def _encode(model, pixel_values):
    return model.encoder(pixel_values, return_dict=True).last_hidden_state


@torch.no_grad()
def _stepwise_cached_vs_full(model, enc, n_steps=N_STEPS):
    """Greedy loop: cached single-token steps vs full-prefix re-forward.

    At every step the last-position logits of the cached path must match the
    logits of re-running the whole prefix without cache.
    """
    tok = make_decoder_input_ids(model, batch_size=enc.shape[0])
    prefix = tok
    cache = None
    for _ in range(n_steps):
        out = model.decoder(
            input_ids=tok,
            encoder_hidden_states=enc,
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        cached_logits = out.logits[:, -1, :]

        full = model.decoder(
            input_ids=prefix, encoder_hidden_states=enc, use_cache=False
        )
        full_logits = full.logits[:, -1, :]

        torch.testing.assert_close(cached_logits, full_logits, atol=ATOL, rtol=RTOL)
        assert torch.equal(cached_logits.argmax(-1), full_logits.argmax(-1))

        tok = cached_logits.argmax(-1, keepdim=True)
        prefix = torch.cat([prefix, tok], dim=1)


def test_cached_vs_uncached_tokens_identical(tiny_model, pixel_values):
    kwargs = dict(pixel_values=pixel_values, max_new_tokens=12, do_sample=False)
    seq_cached = tiny_model.generate(**kwargs, use_cache=True)
    seq_uncached = tiny_model.generate(**kwargs, use_cache=False)
    assert torch.equal(seq_cached, seq_uncached)


def test_stepwise_logits_match_full_reforward(tiny_model, pixel_values):
    enc = _encode(tiny_model, pixel_values)
    _stepwise_cached_vs_full(tiny_model, enc)


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_stepwise_logits_per_decoder_impl(tiny_model, pixel_values, impl):
    tiny_model.decoder.config._attn_implementation = impl
    enc = _encode(tiny_model, pixel_values)
    _stepwise_cached_vs_full(tiny_model, enc)


def test_scores_match_between_cached_and_uncached(tiny_model, pixel_values):
    """Per-step scores must match, not just the argmax-decoded tokens."""
    kwargs = dict(
        pixel_values=pixel_values,
        max_new_tokens=8,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    out_cached = tiny_model.generate(**kwargs, use_cache=True)
    out_uncached = tiny_model.generate(**kwargs, use_cache=False)
    assert torch.equal(out_cached.sequences, out_uncached.sequences)
    for cached, uncached in zip(out_cached.scores, out_uncached.scores):
        torch.testing.assert_close(cached, uncached, atol=ATOL, rtol=RTOL)
