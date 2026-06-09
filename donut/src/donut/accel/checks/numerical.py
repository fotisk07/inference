"""Numerical accuracy checks using synthetic tensors.

No real images or dataset downloads required. Checks that each acceleration
backend produces encoder hidden states and decoder token sequences that are
numerically close to (or identical to) the eager baseline.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import DonutProcessor, VisionEncoderDecoderModel


def _make_inputs(
    model: VisionEncoderDecoderModel,
    processor: DonutProcessor,
    batch_size: int = 1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pixel_values, decoder_input_ids) on model.device with synthetic data."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    img_size = model.encoder.config.image_size
    H, W = (img_size, img_size) if isinstance(img_size, int) else img_size
    pixel_values = torch.randn(
        batch_size,
        3,
        H,
        W,
        dtype=model.dtype,
        device=next(model.parameters()).device,
        generator=gen,
    )
    bos_id = processor.tokenizer.bos_token_id or processor.tokenizer.cls_token_id
    decoder_input_ids = torch.full(
        (batch_size, 1),
        bos_id,
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    return pixel_values, decoder_input_ids


@contextmanager
def _eager_encoder(model: VisionEncoderDecoderModel):
    """Temporarily restore original eager forward on SDPA-patched encoder blocks.

    If SDPA is applied, each block stores _original_forward. This context manager
    swaps it back for the duration, allowing a fair eager vs. accel comparison
    without loading a second model.
    """
    swapped = []
    for stage in model.encoder.encoder.layers:
        for block in stage.blocks:
            sa = block.attention.self
            if hasattr(sa, "_sdpa_patched"):
                swapped.append((sa, sa.forward))
                sa.forward = sa._original_forward
    try:
        yield
    finally:
        for sa, fwd in swapped:
            sa.forward = fwd


def check_encoder_accuracy(
    model: VisionEncoderDecoderModel,
    processor: DonutProcessor,
    *,
    batch_size: int = 1,
    mean_tol: float = 0.05,
    p99_tol: float = 1.0,
    seed: int = 42,
) -> dict:
    """Compare encoder output with SDPA patch vs eager baseline.

    Uses _eager_encoder to temporarily restore original forward, producing a
    genuine eager vs. accelerated comparison without loading a second model.
    Returns error statistics dict.
    """
    pixel_values, _ = _make_inputs(model, processor, batch_size=batch_size, seed=seed)

    with torch.no_grad():
        with _eager_encoder(model):
            enc_eager = model.encoder(pixel_values, return_dict=True)
        enc_accel = model.encoder(pixel_values, return_dict=True)

    abs_err = (enc_eager.last_hidden_state - enc_accel.last_hidden_state).abs()
    n = abs_err.numel()
    max_ae = abs_err.max().item()
    mean_ae = abs_err.mean().item()
    p99_ae = abs_err.flatten().kthvalue(max(1, int(n * 0.99))).values.item()
    ok = mean_ae < mean_tol and p99_ae < p99_tol

    return {"max_ae": max_ae, "mean_ae": mean_ae, "p99_ae": p99_ae, "ok": ok}


def check_decoder_accuracy(
    model: VisionEncoderDecoderModel,
    processor: DonutProcessor,
    *,
    batch_size: int = 1,
    n_tokens: int = 5,
    seed: int = 42,
) -> dict:
    """Check that current backend produces the same token sequence as eager decoder.

    Generates n_tokens tokens with the decoder, comparing the accelerated decoder
    against eager. Uses synthetic pixel_values — token content is arbitrary but
    deterministic (same seed → same result). If first n_tokens match, the
    decoder acceleration is correct.
    """
    pixel_values, decoder_input_ids = _make_inputs(
        model, processor, batch_size=batch_size, seed=seed
    )
    saved_impl = model.decoder.config._attn_implementation

    def _generate(attn_impl: str) -> list[str]:
        model.decoder.config._attn_implementation = attn_impl
        with torch.no_grad():
            enc_out = model.encoder(pixel_values, return_dict=True)
            seqs = model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=enc_out,
                max_new_tokens=n_tokens,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
            ).sequences
        return processor.batch_decode(seqs, skip_special_tokens=True)

    eager_out = _generate("eager")
    accel_out = _generate(saved_impl)
    model.decoder.config._attn_implementation = saved_impl

    exact_match = sum(a == b for a, b in zip(eager_out, accel_out))
    ok = exact_match == batch_size

    return {
        "exact_match": exact_match,
        "n_sequences": batch_size,
        "n_tokens": n_tokens,
        "ok": ok,
    }


def run_accuracy_suite(
    model: VisionEncoderDecoderModel,
    processor: DonutProcessor,
    *,
    batch_size: int = 1,
    save_path: str | None = None,
) -> dict:
    """Run encoder and decoder accuracy checks for the model's current backend.

    Returns a dict with 'encoder' and 'decoder' sub-dicts. Saves to JSON if
    save_path is given (parent dirs created automatically).
    """
    results = {
        "encoder": check_encoder_accuracy(model, processor, batch_size=batch_size),
        "decoder": check_decoder_accuracy(model, processor, batch_size=batch_size),
    }

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, indent=2))

    return results
