import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

from donut.accel.mask_cache import apply_mask_cache


def load_model(model_id: str, device: str) -> tuple:
    """Load model and processor, move to device, set eval mode."""
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id, dtype=torch.bfloat16)
    model.to(device).eval()
    return model, processor


def apply_patch(model, device: str, no_patch: bool) -> str:
    """Apply the attn_mask cache patch. Returns a label for logging."""
    if no_patch:
        return "DISABLED (--no-patch)"
    apply_mask_cache(model)  # auto-selects GPU-direct or CPU-float32 variant
    return "applied (gpu direct)" if device == "cuda" else "applied (cpu float32)"


def apply_accel(model, device: str, backend: str, no_patch: bool) -> str:
    """Apply mask patch and optional attention backend. Returns a label for logging."""
    label = apply_patch(model, device, no_patch)
    if backend == "eager":
        return f"mask={label}, attn=eager"
    from donut.accel.decoder_fa import apply_decoder_fa
    from donut.accel.decoder_sdpa import apply_decoder_sdpa
    from donut.accel.encoder_sdpa import apply_encoder_sdpa

    if backend == "sdpa":
        apply_encoder_sdpa(model)
        apply_decoder_sdpa(model)
        return f"mask={label}, attn=sdpa"
    if backend == "fa2":
        apply_encoder_sdpa(model)
        apply_decoder_fa(model)
        return f"mask={label}, encoder=sdpa, decoder=fa"
    raise ValueError(f"Unknown backend: {backend!r}. Choose: eager, sdpa, fa2")
