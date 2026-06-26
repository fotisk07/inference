import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

from donut.accel import apply_accel, _init_legacy_baseline
from donut.constants import MODEL_ID, TASK_TOKEN


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


def set_processor_image_size(
    processor: DonutProcessor, height: int, width: int
) -> None:
    """Set the resolution the image processor resizes real input images to."""
    processor.image_processor.size = {"height": height, "width": width}


def set_image_size(model, processor, height: int, width: int) -> None:
    """Set the input resolution on BOTH sinks at once.

    The processor (resizes real images) and the encoder config (the model's
    record of its input size) are two independent places the resolution lives;
    writing only one leaves a saved checkpoint internally inconsistent. This is
    the single entry point training/inference should use so they never drift.
    """
    set_processor_image_size(processor, height, width)
    set_encoder_image_size(model, height, width)


# ── Shift tokens (pad + decoder_start) ────────────────────────────────────────
# A teacher-forced labels-forward and generate() both need config.pad_token_id and
# config.decoder_start_token_id (modeling_vision_encoder_decoder.py). donut-base
# carries neither at the top level, so they must be set explicitly — from the
# tokenizer when fine-tuning, or from the decoder sub-config for an untrained run.


def set_donut_shift_tokens(model, processor) -> None:
    """Training: decoder start = the task token, pad = the tokenizer's pad.

    The processor's tokenizer is the authoritative vocab source, so the canonical
    Donut convention (task token is the decoder start) is read straight from it.
    """
    set_shift_tokens(
        model,
        processor.tokenizer.pad_token_id,
        processor.tokenizer.convert_tokens_to_ids(TASK_TOKEN),
    )


def init_shift_tokens_from_decoder(model) -> None:
    """Untrained/bench: seed pad + decoder_start from the MBart decoder sub-config.

    donut-base's decoder sub-config always carries pad_token_id and bos_token_id
    (the top-level config does not). The exact ids are irrelevant to timing; we
    only need valid ints so a forward / generate runs.
    """
    decoder = model.config.decoder
    set_shift_tokens(model, decoder.pad_token_id, decoder.bos_token_id)


# ── Config getters ────────────────────────────────────────────────────────────
# The only places that read these HF-config attributes. Each is a plain read: the
# value is guaranteed present by construction (encoder/decoder sub-config) or by a
# prior set_shift_tokens call, so no defensive getattr is needed.


def encoder_image_size(model) -> tuple[int, int]:
    """(height, width) the encoder expects — always stored as a 2-list."""
    height, width = model.encoder.config.image_size
    return height, width


def encoder_num_channels(model) -> int:
    return model.encoder.config.num_channels


def decoder_vocab_size(model) -> int:
    return model.config.decoder.vocab_size


def decoder_start_token_id(model) -> int:
    return model.config.decoder_start_token_id


def pad_token_id(model) -> int:
    return model.config.pad_token_id


def decoder_start_ids(model, batch_size: int = 1) -> torch.Tensor:
    """A (batch_size, 1) tensor of decoder start tokens on the model's device."""
    return torch.full(
        (batch_size, 1),
        decoder_start_token_id(model),
        dtype=torch.long,
        device=next(model.parameters()).device,
    )


def load_baseline_model(
    model_id: str = MODEL_ID,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    *,
    tiny: bool = False,
) -> tuple[VisionEncoderDecoderModel, str]:
    """Load a model with NO accelerations applied

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
