import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

from donut.accel import apply_accel, _init_legacy_baseline
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
    model.to(device).eval()  # ty: ignore[invalid-argument-type]
    _init_legacy_baseline(model)
    apply_accel(model, backend)

    # The pretrained checkpoint's generation_config carries a stale max_length
    # (20) left over from pretraining. Every caller in this codebase controls
    # length via max_new_tokens, so clear it to avoid the "both max_new_tokens
    # and max_length seem to have been set" warning on every generate() call.
    model.generation_config.max_length = None

    return model, processor


def autocast(device: str, precision: str):
    """bf16 autocast on CUDA when precision=="bf16"; an inert no-op otherwise.

    The single source for "how to run the model's forward in mixed precision",
    shared by the training loop and the training-step bench so they never drift.
    """
    enabled = device.startswith("cuda") and precision == "bf16"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return torch.autocast(
        device_type=device_type, dtype=torch.bfloat16, enabled=enabled
    )


# ── Config pokes ──────────────────────────────────────────────────────────────
# Small named wrappers around the handful of HF config mutations needed to make
# Donut do field extraction. Each is a one-liner; the point is that the call site
# reads as intent ("set the image size") instead of an opaque attribute assignment.


def fit_decoder_to_vocab(model, processor) -> None:
    """Grow the decoder's token embeddings to the processor's (extended) vocab."""
    model.decoder.resize_token_embeddings(len(processor.tokenizer))


def set_shift_tokens(model, pad_token_id: int, decoder_start_token_id: int) -> None:
    """Set the two ids a teacher-forced labels-forward needs.

    shift_tokens_right (modeling_vision_encoder_decoder.py) reads
    config.pad_token_id (loss mask / padding) and config.decoder_start_token_id
    (auto-prepended to the labels as the decoder input).
    """
    model.config.pad_token_id = pad_token_id
    model.config.decoder_start_token_id = decoder_start_token_id


def set_encoder_image_size(model, height: int, width: int) -> None:
    """Set the encoder input resolution — drives the synthetic pixel_values shape."""
    model.encoder.config.image_size = [height, width]


def load_baseline_model(
    model_id: str = MODEL_ID,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    *,
    tiny: bool = False,
) -> tuple[VisionEncoderDecoderModel, str]:
    """Load a model with NO accelerations applied — the single source of truth for
    the benchmarking scripts, which apply/revert each backend themselves.

    tiny=True returns the offline random fixture (donut.synthetic.make_tiny_model,
    no download, id "tiny-random-donut"); otherwise the real checkpoint loaded with
    backend="baseline". device/dtype default the same way as load_model. Returns
    (model, model_id).
    """
    if tiny:
        from donut.synthetic import make_tiny_model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        model = make_tiny_model(seed=0).to(device=device, dtype=dtype)
        return model.eval(), "tiny-random-donut"
    model, _ = load_model(model_id, device, dtype, backend="baseline")
    return model.eval(), model_id
