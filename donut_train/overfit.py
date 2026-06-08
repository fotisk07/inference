"""
Overfit sanity check for donut fine-tuning.

Trains on a single image for MAX_STEPS steps and checks that the loss collapses.
If loss does not drop below PASS_THRESHOLD in 300 steps, something is broken.

Run: cd donut_train && uv run python overfit.py
"""

import json
from pathlib import Path

import torch
from dataset import TASK_TOKEN, build_processor, format_label
from PIL import Image
from train import build_model

# ── Config ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
MODEL_NAME = "naver-clova-ix/donut-base"
IMAGE_PATH = _HERE / "../test_data/images/train/test_data.jpg"
ANNOTATION_PATH = _HERE / "../test_data/new_cardxie_annotations/train/test_data.json"

MAX_TARGET_LENGTH = 128
LR = 5e-4
MAX_STEPS = 300
DECODE_EVERY = 25
PASS_THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    print(f"\n{'=' * 60}")
    print("  Donut overfit sanity check")
    print(
        f"  Device : {DEVICE}  LR={LR}  Steps={MAX_STEPS}  Threshold={PASS_THRESHOLD}"
    )
    print(f"{'=' * 60}\n")

    # --- setup ---
    print(f"Loading processor and model from {MODEL_NAME} ...")
    processor = build_processor(MODEL_NAME)
    model = build_model(processor, MODEL_NAME).to(DEVICE)
    print(f"Vocab size: {len(processor.tokenizer)}\n")

    # --- data ---
    image = Image.open(IMAGE_PATH).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)

    with open(ANNOTATION_PATH) as f:
        annotation = json.load(f)

    target_text = (
        TASK_TOKEN + format_label(annotation["fields"]) + processor.tokenizer.eos_token
    )
    tokenized = processor.tokenizer(
        target_text,
        add_special_tokens=False,
        max_length=MAX_TARGET_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenized.input_ids.squeeze(0).clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels_batch = labels.unsqueeze(0).to(DEVICE)

    real_tokens = (labels != -100).sum().item()
    token_strings = processor.tokenizer.convert_ids_to_tokens(
        tokenized.input_ids.squeeze(0)[:real_tokens].tolist()
    )
    print(f"  Image       : {IMAGE_PATH.name}  {image.size}")
    print(f"  Target text : {target_text!r}")
    print(f"  Token count : {real_tokens}")
    print(f"  Tokens      : {token_strings}\n")

    # --- training loop ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    print(f"{'─' * 60}")
    print(f"  Training {MAX_STEPS} steps  (decoding every {DECODE_EVERY})")
    print(f"{'─' * 60}")

    model.train()
    last_loss = float("inf")

    for step in range(MAX_STEPS):
        optimizer.zero_grad()
        loss = model(pixel_values=pixel_values, labels=labels_batch).loss
        loss.backward()
        optimizer.step()

        last_loss = loss.item()
        print(f"  Step {step + 1:3d}/{MAX_STEPS}  loss={last_loss:.4f}")

        if (step + 1) % DECODE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                gen_ids = model.generate(
                    pixel_values=pixel_values,
                    decoder_start_token_id=processor.tokenizer.pad_token_id,
                    max_new_tokens=64,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            predicted = processor.tokenizer.decode(
                gen_ids[0][1:], skip_special_tokens=False
            )
            print(f"\n  ┌─ Decode @ step {step + 1} {'─' * 38}")
            print(f"  │  Target : {target_text!r}")
            print(f"  │  Predict: {predicted!r}")
            print(
                f"  └─ {'EXACT MATCH' if predicted.strip() == target_text.strip() else 'no match yet'}\n"
            )
            model.train()

    # --- result ---
    print(f"{'=' * 60}")
    print(f"  Final loss : {last_loss:.4f}")
    if last_loss < PASS_THRESHOLD:
        print(f"  PASS — model memorized the sample (loss < {PASS_THRESHOLD})")
    else:
        print(f"  FAIL — loss did not reach threshold {PASS_THRESHOLD}")
        print(
            "         Check token list above: each special token should be a single piece."
        )
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
