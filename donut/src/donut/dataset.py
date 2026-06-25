import json
import re
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from transformers import DonutProcessor

from donut.constants import DEFAULT_MAX_LENGTH, FIELD_TOKENS, MISSING_TOKEN, TASK_TOKEN

### Vocabulary ###########################################################


def _t2j_pairs() -> list[tuple[str, str]]:
    """Return (open, close) token pairs for token2json format."""
    return [(f"<s_{tok[1:-1]}>", f"</s_{tok[1:-1]}>") for tok in FIELD_TOKENS]


### Processor ###########################################################


def register_field_tokens(processor: DonutProcessor) -> DonutProcessor:
    """Register the task token + token2json field tokens onto a DonutProcessor.

    Mutates `processor` in place (and returns it) so a single processor — the one
    load_model() already built — is the only source of truth for the vocabulary.
    """
    extra = [t for pair in _t2j_pairs() for t in pair] + [MISSING_TOKEN]
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [TASK_TOKEN] + extra}
    )
    return processor


### Label encoding ###########################################################


def format_label(fields: list[dict]) -> str:
    """
    Convert annotation fields to a token2json-format string.

    Derives the field name from the last segment of field_name (after '/'),
    keeping only the first occurrence when duplicates appear. XML-like and
    parseable with processor.token2json():
        present : "<s_E-mail>foo@bar.com</s_E-mail>"
        missing : "<s_E-mail><missing></s_E-mail>"

    Every field in FIELD_TOKENS order is emitted — present or not — so the model
    learns a fixed-length, fixed-order template with an explicit "absent" signal.
    """
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
    for open_tok, close_tok in _t2j_pairs():
        leaf = open_tok[3:-1]  # "<s_E-mail>" → "E-mail"
        value = values.get(leaf, "")
        parts.append(f"{open_tok}{value or MISSING_TOKEN}{close_tok}")
    return "".join(parts)


### Label decoding #######################################################


def _flatten(obj, prefix: str = "") -> dict[str, str]:
    """Recursively flatten a nested dict returned by processor.token2json()."""
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, k))
    elif isinstance(obj, list):
        for item in obj:
            out.update(_flatten(item, prefix))
    else:
        if prefix:
            value = str(obj) if obj is not None else ""
            out[prefix] = "" if value == MISSING_TOKEN else value
    return out


def _parse_token2json(decoded: str, processor: DonutProcessor) -> dict[str, str]:
    """
    Parse token2json-format output into a {field: value} dict.

    The model produces something like:
        <s_donut><s_E-mail>a@b.com</s_E-mail><s_valor_da_nota>100</s_valor_da_nota>...

    We strip the task token then delegate to processor.token2json().
    """
    text = re.sub(re.escape(TASK_TOKEN), "", decoded, count=1)
    try:
        parsed = processor.token2json(text)
    except Exception:
        parsed = {}
    return _flatten(parsed)


def parse_prediction(decoded: str, processor: DonutProcessor) -> dict[str, str]:
    """Decode a raw model output string into a {field_name: value} dict."""
    return _parse_token2json(decoded, processor)


### Data loading #####################################################


def load_samples(split_json: Path) -> list[dict]:
    split_json = Path(split_json)
    base = split_json.parent
    with open(split_json) as f:
        records = json.load(f)
    return [{"image": base / r["image"], "fields": r["fields"]} for r in records]


### Dataset ########################################################


class DonutDataset(Dataset):
    """
    Args:
        samples:    list of {"image": Path, "fields": [...]} dicts from load_samples()
        processor:  DonutProcessor from build_processor() — must have special tokens registered
        max_length: maximum decoder sequence length (default 128)

    Each item returns:
        pixel_values  (3, H, W) float tensor
        labels        (max_length,) int64 tensor, padding replaced with -100
    """

    def __init__(
        self,
        samples: list[dict],
        processor: DonutProcessor,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length

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

        # Labels: fields then EOS. The TASK_TOKEN is the decoder_start_token_id
        # (set in train.build_model), which shift_tokens_right auto-prepends as the
        # decoder input — so it must NOT also appear here as a target.
        label_text = format_label(sample["fields"]) + self.processor.tokenizer.eos_token

        tokenized = self.processor.tokenizer(
            label_text,
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
        }
