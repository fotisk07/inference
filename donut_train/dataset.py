"""
Data loading and preprocessing for donut fine-tuning.

Designed to be importable in notebooks:
    from dataset import DonutDataset, build_processor
    ds = DonutDataset(samples, processor)
    print(ds[0]["target_text"])   # human-readable label
"""

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from transformers import DonutProcessor

# ── Token vocabulary ──────────────────────────────────────────────────────────
TASK_TOKEN = "<s_donut>"
FIELD_TOKENS = [
    "<destinataire>",
    "<E-mail>",
    "<cpf_cnpj_prestador>",
    "<cpf_cnpj_tomador>",
    "<data_emissao>",
    "<numero_da_nota>",
    "<servico_prestado>",
    "<valor_da_nota>",
]
ALL_SPECIAL_TOKENS = [TASK_TOKEN] + FIELD_TOKENS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def build_processor(model_name: str) -> DonutProcessor:
    """Load DonutProcessor and register all task + field tokens."""
    processor = DonutProcessor.from_pretrained(model_name)
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": ALL_SPECIAL_TOKENS}
    )
    return processor


def format_label(fields: list[dict], include_missing: bool = False) -> str:
    """
    Converts annotation fields to a token-wrapped string.

    Derives the token name from the last segment of field_name (after '/'),
    so it's robust to prefix typos in the annotation paths.

    Duplicate field_names keep only the first occurrence.

    include_missing=True: absent/empty fields appear as <token><token> so the
    model explicitly learns "this field is not present" rather than never seeing
    a signal for missing fields.

    Examples:
        present : "<E-mail> foo@bar.com <E-mail>"
        missing : "<E-mail><E-mail>"   (only when include_missing=True)
    """
    # deduplicate: keep first occurrence of each field_name
    seen: set[str] = set()
    deduped = []
    for f in fields:
        if f["field_name"] not in seen:
            seen.add(f["field_name"])
            deduped.append(f)

    values = {
        f["field_name"].split("/")[-1]: f.get("annotator_text", "").strip()
        for f in deduped
    }

    # iterate in canonical FIELD_TOKENS order for deterministic output
    parts = []
    for token in FIELD_TOKENS:
        leaf = token[1:-1]  # "<E-mail>" → "E-mail"
        value = values.get(leaf, "")
        if value:
            parts.append(f"{token} {value} {token}")
        elif include_missing:
            parts.append(f"{token}{token}")

    return " ".join(parts)


def load_samples(images_dir: Path, annotations_dir: Path) -> list[dict]:
    """
    Pairs each image in images_dir with its matching JSON annotation.
    Returns: [{"image": Path, "fields": [...]}, ...]
    Skips images with no matching annotation.
    """
    samples = []
    for img_path in sorted(Path(images_dir).iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        ann_path = Path(annotations_dir) / (img_path.stem + ".json")
        if not ann_path.exists():
            continue
        with open(ann_path) as f:
            annotation = json.load(f)
        samples.append({"image": img_path, "fields": annotation["fields"]})
    return samples


class DonutDataset(Dataset):
    """
    Args:
        samples:    list of {"image": Path, "fields": [...]} dicts from load_samples()
        processor:  DonutProcessor from build_processor() — must have special tokens registered
        max_length: maximum decoder sequence length (default 128)

    Each item returns:
        pixel_values  (3, H, W) float tensor
        labels        (max_length,) int64 tensor, padding replaced with -100
        target_text   str — the raw label string, useful for notebook inspection
    """

    def __init__(
        self,
        samples: list[dict],
        processor: DonutProcessor,
        max_length: int = 128,
        include_missing: bool = False,
    ):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.include_missing = include_missing

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        img = sample["image"]
        image = (Image.open(img) if isinstance(img, (str, Path)) else img).convert(
            "RGB"
        )
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(
            0
        )

        # Labels: TASK_TOKEN first (naver-standard), then fields, then EOS
        target_text = (
            TASK_TOKEN
            + format_label(sample["fields"], self.include_missing)
            + self.processor.tokenizer.eos_token
        )

        tokenized = self.processor.tokenizer(
            target_text,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.squeeze(0).clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "target_text": target_text,
        }
