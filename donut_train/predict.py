"""Generate predictions from a fine-tuned Donut checkpoint."""

import json
from pathlib import Path

import torch
from dataset import TASK_TOKEN, load_samples, parse_prediction
from tqdm import tqdm
from donut import load_model
from metrics import summarize
from PIL import Image
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)

    checkpoint: str  # required; path to a checkpoint dir saved by train.py (e.g. checkpoints/best)

    # Input data — same aggregate JSON format as train.py
    data_json: str = "../test_data/train.json"

    # If given, write per-document {image, gt, pred} records to this JSON path.
    output_json: str | None = None

    # Acceleration backend passed to donut.load_model()
    backend: str = "sdpa"

    # Maximum decoder tokens to generate per image
    max_new_tokens: int = 128

    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


# ── Model loading ─────────────────────────────────────────────────────────────


def load_from_checkpoint(ckpt_dir: str, backend: str, device: str):
    """
    Load model and processor from a checkpoint dir saved by train.save_checkpoint.

    from_pretrained on the local dir restores the fine-tuned weights, the
    added-token processor, image_size, and decoder_start — everything
    save_pretrained persists. token2json_format (which it doesn't) is read from
    the train_meta.json sidecar to pick the right output parser.
    """
    ckpt_dir = Path(ckpt_dir)
    meta = json.loads((ckpt_dir / "train_meta.json").read_text())
    token2json_format = meta["token2json_format"]

    model, processor = load_model(
        model_id=str(ckpt_dir), device=device, backend=backend
    )
    model.eval()

    return model, processor, token2json_format


# ── Inference loop ────────────────────────────────────────────────────────────


def run_predictions(
    model,
    processor,
    samples: list[dict],
    *,
    token2json_format: bool,
    device: str,
    max_new_tokens: int = 128,
    progress: bool = True,
) -> list[dict]:
    """Generate field predictions for every sample.

    Returns [{"image", "gt": {field: value}, "pred": {field: value}}] — used both
    for metrics.py scoring and (verbatim) as the saved --output_json.
    """
    model.eval()
    model_dtype = next(model.parameters()).dtype
    # Canonical Donut: the task token is the decoder start (train.build_model sets
    # decoder_start_token_id = <s_donut>), so seeding generation with it matches the
    # training-time decoder input position-for-position.
    decoder_input_ids = processor.tokenizer(
        TASK_TOKEN, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    results = []
    iterator = tqdm(samples, desc="predicting") if progress else samples
    for sample in iterator:
        image = Image.open(sample["image"]).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(
            device=device, dtype=model_dtype
        )

        with torch.no_grad():
            output_ids = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=max_new_tokens,
            )

        decoded = processor.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        pred_fields = parse_prediction(decoded, token2json_format, processor)

        # Ground truth — may be absent (inference-only mode)
        gt_fields = {
            f["field_name"].split("/")[-1]: f.get("annotator_text", "").strip()
            for f in sample.get("fields", [])
            if f.get("annotator_text", "").strip()
        }

        results.append(
            {"image": str(sample["image"]), "gt": gt_fields, "pred": pred_fields}
        )

    return results


def predict(cfg: Config) -> None:
    print(f"Checkpoint : {cfg.checkpoint}")
    print(f"Data       : {cfg.data_json}")
    print(f"Backend    : {cfg.backend}  device={cfg.device}\n")

    model, processor, token2json_format = load_from_checkpoint(
        cfg.checkpoint, cfg.backend, cfg.device
    )

    samples = load_samples(Path(cfg.data_json))
    print(f"Processing {len(samples)} samples ...")

    results = run_predictions(
        model,
        processor,
        samples,
        token2json_format=token2json_format,
        device=cfg.device,
        max_new_tokens=cfg.max_new_tokens,
    )

    # Metrics — only when GT is present
    if any(r["gt"] for r in results):
        summarize(results, soft=False)
        summarize(results, soft=True)

    # Save output JSON — per-document {image, gt, pred} records
    if cfg.output_json:
        out_path = Path(cfg.output_json)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    predict(Config())
