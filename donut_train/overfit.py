"""
Overfit sanity check for donut fine-tuning.

Trains on a single image for MAX_STEPS steps and checks that the loss collapses.
If loss does not drop below PASS_THRESHOLD (~0.5) in 300 steps, something is broken
in the data pipeline, tokenisation, or model configuration.

Task token design (naver-standard):
  - <s_donut> is the FIRST token in the label sequence
  - decoder_start_token_id = pad_token_id
  - Teacher-forced input: [pad, <s_donut>, <field>, ...]
  - Model must predict <s_donut> as its first output
  This is the correct approach for multi-task setups and matches the original donut repo.

Run: cd donut_train && uv run python overfit.py
"""

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# ── Config ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
MODEL_NAME = "naver-clova-ix/donut-base"
IMAGE_PATH = _HERE / "../test_data/images/train/test_data.jpg"
ANNOTATION_PATH = _HERE / "../test_data/new_cardxie_annotations/train/test_data.json"

TASK_TOKEN = "<s_donut>"
SPECIAL_TOKENS = [
    "<s_donut>",
    "<destinataire>",
    "<E-mail>",
    "<cpf_cnpj_prestador>",
    "<cpf_cnpj_tomador>",
    "<data_emissao>",
    "<numero_da_nota>",
    "<servico_prestado>",
    "<valor_da_nota>",
]

MAX_TARGET_LENGTH = 128
LR = 5e-4
MAX_STEPS = 300
DECODE_EVERY = 25
PASS_THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Inlined label formatter ───────────────────────────────────────────────────
def format_label(fields: list[dict]) -> str:
    """Produces '<leaf_token> value <leaf_token>' for each non-empty field."""
    parts = []
    for f in fields:
        value = f.get("annotator_text", "").strip()
        if value:
            leaf = f["field_name"].split("/")[-1]
            parts.append(f"<{leaf}> {value} <{leaf}>")
    return " ".join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"\n{'='*60}")
    print(f"  Donut overfit sanity check")
    print(f"  Device : {DEVICE}")
    print(f"  LR     : {LR}   Max steps: {MAX_STEPS}   Pass threshold: {PASS_THRESHOLD}")
    print(f"{'='*60}\n")

    # ── Load processor + model ────────────────────────────────────────────────
    print(f"Loading processor and model from {MODEL_NAME} ...")
    processor = DonutProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    vocab_before = len(processor.tokenizer)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    vocab_after = len(processor.tokenizer)
    model.decoder.resize_token_embeddings(vocab_after)
    print(f"Vocab: {vocab_before} → {vocab_after}  (+{vocab_after - vocab_before} tokens)")

    # naver-standard: pad as decoder start, task token lives in labels
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.pad_token_id

    model = model.to(DEVICE)
    print(f"Model on {DEVICE}.\n")

    # ── Prepare data ──────────────────────────────────────────────────────────
    print("Preparing data ...")
    image = Image.open(IMAGE_PATH).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)

    with open(ANNOTATION_PATH) as f:
        annotation = json.load(f)

    field_text = format_label(annotation["fields"])
    target_text = TASK_TOKEN + field_text + processor.tokenizer.eos_token

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

    # Count real (non-padding) tokens
    real_tokens = (labels != -100).sum().item()
    token_strings = processor.tokenizer.convert_ids_to_tokens(
        tokenized.input_ids.squeeze(0)[:real_tokens].tolist()
    )

    print(f"  Image         : {IMAGE_PATH.name}  {image.size}")
    print(f"  Target text   : {target_text!r}")
    print(f"  Token count   : {real_tokens}  (excl. padding)")
    print(f"  Tokens        : {token_strings}")

    # Warn if any special token got split (would appear as multiple pieces)
    for st in SPECIAL_TOKENS:
        tid = processor.tokenizer.convert_tokens_to_ids(st)
        if tid == processor.tokenizer.unk_token_id:
            print(f"  WARNING: {st!r} resolved to UNK — registration may have failed")

    print()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"{'─'*60}")
    print(f"  Training for {MAX_STEPS} steps  (decoding every {DECODE_EVERY} steps)")
    print(f"{'─'*60}")

    model.train()
    last_loss = float("inf")

    for step in range(MAX_STEPS):
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels_batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        last_loss = loss.item()
        print(f"  Step {step+1:3d}/{MAX_STEPS}  loss={last_loss:.4f}")

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
            predicted = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=False)
            print(f"\n  ┌─ Decode @ step {step+1} {'─'*38}")
            print(f"  │  Target : {target_text!r}")
            print(f"  │  Predict: {predicted!r}")
            match = predicted.strip() == target_text.strip()
            print(f"  └─ {'EXACT MATCH ✓' if match else 'no match yet'}\n")
            model.train()

    # ── Result ────────────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"  Final loss : {last_loss:.4f}")
    if last_loss < PASS_THRESHOLD:
        print(f"  PASS — model memorized the sample (loss < {PASS_THRESHOLD})")
    else:
        print(f"  FAIL — loss {last_loss:.4f} did not reach threshold {PASS_THRESHOLD}")
        print(f"         Check the 'Tokens' line above: special tokens should appear")
        print(f"         as single pieces, not split into sub-tokens.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
