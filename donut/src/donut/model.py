import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

from donut.accel import apply_accel
from donut.constants import MODEL_ID


def load_model(
    model_id: str = MODEL_ID,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    backend: str = "sdpa",
) -> tuple[VisionEncoderDecoderModel, DonutProcessor]:
    """Load Donut model and processor with the specified acceleration backend.

    device=None picks cuda when available, else cpu; dtype=None picks bfloat16
    on cuda, float32 on cpu. Applies mask caching and the chosen attention
    backend before returning. Returns (model, processor) ready for inference.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id, dtype=dtype)
    model.to(device).eval()
    apply_accel(model, backend)

    # The pretrained checkpoint's generation_config carries a stale max_length
    # (20) left over from pretraining. Every caller in this codebase controls
    # length via max_new_tokens, so clear it to avoid the "both max_new_tokens
    # and max_length seem to have been set" warning on every generate() call.
    model.generation_config.max_length = None

    return model, processor
