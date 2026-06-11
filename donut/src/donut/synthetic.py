"""Synthetic model inputs and a tiny offline model for tests, audits, benchmarks.

No real images, processor, or dataset downloads required — shapes and special
token ids are derived from the model config, so these helpers work for both
the real Donut checkpoint and the tiny randomly-initialized fixture below.
"""

import torch

# Same architecture family as the real checkpoint (DonutSwin + MBart) but ~69K
# params: instant to build, runs on CPU, needs no downloads. Final encoder
# hidden size (embed_dim * 2**(len(depths)-1) = 32) matches d_model, so no
# enc-to-dec projection — same as the real model.
TINY_ENCODER = dict(
    image_size=64,
    patch_size=4,
    num_channels=3,
    embed_dim=16,
    depths=[2, 2],
    num_heads=[2, 4],
    window_size=4,
)
TINY_DECODER = dict(
    vocab_size=120,
    d_model=32,
    decoder_layers=2,
    decoder_attention_heads=4,
    decoder_ffn_dim=64,
    max_position_embeddings=128,
    pad_token_id=1,
    bos_token_id=0,
    eos_token_id=2,
)


def make_tiny_model(seed: int = 0):
    """Randomly-initialized tiny VisionEncoderDecoderModel (CPU, float32)."""
    from transformers import (
        DonutSwinConfig,
        MBartConfig,
        VisionEncoderDecoderConfig,
        VisionEncoderDecoderModel,
    )

    cfg = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        DonutSwinConfig(**TINY_ENCODER), MBartConfig(**TINY_DECODER)
    )
    cfg.decoder_start_token_id = TINY_DECODER["bos_token_id"]
    cfg.pad_token_id = TINY_DECODER["pad_token_id"]
    torch.manual_seed(seed)
    model = VisionEncoderDecoderModel(config=cfg).eval()
    # Random weights give near-flat logits; sharpen lm_head so greedy argmax
    # has clear winners and token-equality checks can't flake on ties.
    model.decoder.lm_head.weight.data *= 8
    return model


def make_pixel_values(model, batch_size: int = 1, seed: int = 42) -> torch.Tensor:
    """Random pixel_values on the model's device/dtype, sized from its config."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    img_size = model.encoder.config.image_size
    h, w = (img_size, img_size) if isinstance(img_size, int) else img_size
    channels = getattr(model.encoder.config, "num_channels", 3)
    pixel_values = torch.randn(batch_size, channels, h, w, generator=gen)
    return pixel_values.to(dtype=model.dtype, device=next(model.parameters()).device)


def make_decoder_input_ids(
    model, batch_size: int = 1, bos_id: int | None = None
) -> torch.Tensor:
    """A (batch_size, 1) tensor of decoder start tokens."""
    if bos_id is None:
        bos_id = model.config.decoder_start_token_id
        if bos_id is None:
            bos_id = model.config.decoder.bos_token_id
    return torch.full(
        (batch_size, 1),
        bos_id,
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
