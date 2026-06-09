import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

from donut.accel import Backend, apply_accel
from donut.constants import MODEL_ID


def load_model(
    model_id: str = MODEL_ID,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    backend: "Backend | str" = Backend.SDPA,
    compile: bool = False,
) -> tuple[VisionEncoderDecoderModel, DonutProcessor]:
    """Load Donut model and processor with the specified acceleration backend.

    Applies mask caching and the chosen attention backend before returning.
    Pass compile=True to additionally wrap with torch.compile(dynamic=True).
    Returns (model, processor) ready for inference.
    """
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device).eval()
    apply_accel(model, backend, compile=compile)
    return model, processor
