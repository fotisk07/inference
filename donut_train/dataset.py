import json
from pathlib import Path

import yaml
from PIL import Image
from torch.utils.data import Dataset
from transformers import DonutProcessor

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(_CONFIG_PATH) as _f:
    _cfg = yaml.safe_load(_f)

TASK_TOKEN: str = _cfg["TASK_TOKEN"]
FIELD_TOKENS: list[str] = _cfg["FIELD_TOKENS"]
ALL_SPECIAL_TOKENS: list[str] = [TASK_TOKEN] + FIELD_TOKENS

# Derive token2json open/close pairs: "<E-mail>" → ("<s_E-mail>", "</s_E-mail>")
FIELD_TOKENS_T2J: list[tuple[str, str]] = [
    (f"<s_{tok[1:-1]}>", f"</s_{tok[1:-1]}>") for tok in FIELD_TOKENS
]
ALL_SPECIAL_TOKENS_T2J: list[str] = [TASK_TOKEN] + [
    t for pair in FIELD_TOKENS_T2J for t in pair
]

IMAGE_EXTENSIONS: set[str] = set(_cfg["IMAGE_EXTENSIONS"])


def build_processor(model_name: str, token2json_format: bool = False) -> DonutProcessor:
    """Load DonutProcessor and register all task + field tokens."""
    processor = DonutProcessor.from_pretrained(model_name)
    tokens = ALL_SPECIAL_TOKENS_T2J if token2json_format else ALL_SPECIAL_TOKENS
    processor.tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    return processor


def format_label(
    fields: list[dict],
    include_missing: bool = False,
    token2json_format: bool = False,
) -> str:
    """
    Converts annotation fields to a token-wrapped string.

    Derives the token name from the last segment of field_name (after '/'),
    so it's robust to prefix typos in the annotation paths.

    Duplicate field_names keep only the first occurrence.

    token2json_format=False (default):
        present : "<E-mail> foo@bar.com <E-mail>"
        missing : "<E-mail><E-mail>"   (only when include_missing=True)

    token2json_format=True — compatible with processor.token2json():
        present : "<s_E-mail>foo@bar.com</s_E-mail>"
        missing : "<s_E-mail></s_E-mail>"   (only when include_missing=True)
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

    parts = []
    if token2json_format:
        for open_tok, close_tok in FIELD_TOKENS_T2J:
            leaf = open_tok[3:-1]  # "<s_E-mail>" → "E-mail"
            value = values.get(leaf, "")
            if value:
                parts.append(f"{open_tok}{value}{close_tok}")
            elif include_missing:
                parts.append(f"{open_tok}{close_tok}")
        return "".join(parts)
    else:
        for token in FIELD_TOKENS:
            leaf = token[1:-1]  # "<E-mail>" → "E-mail"
            value = values.get(leaf, "")
            if value:
                parts.append(f"{token} {value} {token}")
            elif include_missing:
                parts.append(f"{token}{token}")
        return " ".join(parts)


def load_samples(split_json: Path) -> list[dict]:
    split_json = Path(split_json)
    base = split_json.parent
    with open(split_json) as f:
        records = json.load(f)
    return [{"image": base / r["image"], "fields": r["fields"]} for r in records]


class DonutDataset(Dataset):
    """
    Args:
        samples:    list of {"image": Path, "fields": [...]} dicts from load_samples()
        processor:  DonutProcessor from build_processor() — must have special tokens registered
        max_length: maximum decoder sequence length (default 128)

    Each item returns:
        pixel_values  (3, H, W) float tensor
        labels        (max_length,) int64 tensor, padding replaced with -100
        target_text    the raw label string, useful for notebook inspection
    """

    def __init__(
        self,
        samples: list[dict],
        processor: DonutProcessor,
        max_length: int = 128,
        include_missing: bool = False,
        token2json_format: bool = False,
    ):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.include_missing = include_missing
        self.token2json_format = token2json_format

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
            + format_label(
                sample["fields"], self.include_missing, self.token2json_format
            )
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
