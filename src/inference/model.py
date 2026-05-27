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
