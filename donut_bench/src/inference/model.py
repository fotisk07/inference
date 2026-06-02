import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

from inference.patches import patch_attn_mask, patch_attn_mask_gpu


def load_model(model_id: str, device: str) -> tuple:
    """Load model and processor, move to device, set eval mode."""
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    )
    model.to(device).eval()
    return model, processor


def apply_patch(model, device: str, no_patch: bool) -> str:
    """Apply the appropriate attn_mask patch. Returns a label for logging."""
    if no_patch:
        return "DISABLED (--no-patch)"
    if device == "cuda":
        patch_attn_mask_gpu(model)
        return "applied (gpu direct)"
    patch_attn_mask(model)
    return "applied (cpu float32)"


def apply_accel(model, device: str, backend: str, no_patch: bool) -> str:
    """Apply mask patch and optional attention backend. Returns a label for logging."""
    label = apply_patch(model, device, no_patch)
    if backend == "eager":
        return f"mask={label}, attn=eager"
    from inference.accel.sdpa import activate_decoder_sdpa, patch_swin_sdpa
    from inference.accel.fa2 import activate_decoder_fa2

    if backend == "sdpa":
        patch_swin_sdpa(model)
        activate_decoder_sdpa(model)
        return f"mask={label}, attn=sdpa"
    if backend == "fa2":
        patch_swin_sdpa(model)
        activate_decoder_fa2(model)
        return f"mask={label}, encoder=sdpa, decoder=fa2"
    raise ValueError(f"Unknown backend: {backend!r}. Choose: eager, sdpa, fa2")
