"""Generate predictions from a fine-tuned Donut checkpoint."""

import json
from pathlib import Path

import torch
from dataset import TASK_TOKEN, build_processor, load_samples, parse_prediction
from tqdm import tqdm
from donut import load_model
from metrics import summarize
from PIL import Image
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)

    checkpoint: str  # required; path to a .pt file saved by train.py

    # Input data — same aggregate JSON format as train.py
    data_json: str = "../test_data/train.json"

    # If given, write predictions as a JSON file in the same aggregate format as
    # train.json, with annotator_text replaced by the model's predicted value.
    output_json: str | None = None

    # Acceleration backend passed to donut.load_model()
    backend: str = "sdpa"

    # Maximum decoder tokens to generate per image
    max_new_tokens: int = 128

    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


# ── Model loading ─────────────────────────────────────────────────────────────


def load_from_checkpoint(ckpt_path: str, backend: str, device: str):
    """
    Reconstruct model and processor from a checkpoint saved by train.py.

    The checkpoint embeds the metadata (model_name, token2json_format, etc.)
    needed to rebuild the exact same architecture and vocabulary without
    requiring the user to pass those flags again.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_name = ckpt.get("model_name", "naver-clova-ix/donut-base")
    token2json_format = ckpt.get("token2json_format", False)
    image_size = ckpt.get("image_size", (1280, 960))

    processor = build_processor(model_name, token2json_format)
    processor.image_processor.size = {"height": image_size[0], "width": image_size[1]}

    model, _ = load_model(model_id=model_name, device=device, backend=backend)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.pad_token_id
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, processor, token2json_format


# ── Inference loop ────────────────────────────────────────────────────────────


def predict(cfg: Config) -> None:
    print(f"Checkpoint : {cfg.checkpoint}")
    print(f"Data       : {cfg.data_json}")
    print(f"Backend    : {cfg.backend}  device={cfg.device}\n")

    model, processor, token2json_format = load_from_checkpoint(
        cfg.checkpoint, cfg.backend, cfg.device
    )

    samples = load_samples(Path(cfg.data_json))
    print(f"Processing {len(samples)} samples ...")

    model_dtype = next(model.parameters()).dtype
    task_ids = processor.tokenizer(
        TASK_TOKEN, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(cfg.device)

    results = []  # for metrics.py
    output_records = []  # for output JSON

    for i, sample in enumerate(tqdm(samples, desc="predicting")):
        image = Image.open(sample["image"]).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(
            device=cfg.device, dtype=model_dtype
        )

        with torch.no_grad():
            output_ids = model.generate(
                pixel_values,
                decoder_input_ids=task_ids,
                max_new_tokens=cfg.max_new_tokens,
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
            {"image": str(sample["image"]), "pred": pred_fields, "gt": gt_fields}
        )

        # Output record: same aggregate JSON format with predicted values as annotator_text
        output_records.append(
            {
                "image": str(sample["image"]),
                "fields": [
                    {"field_name": k, "annotator_text": v}
                    for k, v in pred_fields.items()
                ],
            }
        )

    # Metrics — only when GT is present
    if any(r["gt"] for r in results):
        summarize(results, soft=False)
        summarize(results, soft=True)

    # Save output JSON
    if cfg.output_json:
        out_path = Path(cfg.output_json)
        out_path.write_text(json.dumps(output_records, indent=2, ensure_ascii=False))
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    predict(Config())
