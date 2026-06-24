import json
import re
from pathlib import Path

import yaml
from PIL import Image
from torch.utils.data import Dataset
from transformers import DonutProcessor

# ── Vocabulary ────────────────────────────────────────────────────────────────


def _load_vocab() -> tuple[str, list[str], str]:
    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    return cfg["TASK_TOKEN"], cfg["FIELD_TOKENS"], cfg["MISSING_TOKEN"]


TASK_TOKEN, FIELD_TOKENS, MISSING_TOKEN = _load_vocab()


def _t2j_pairs() -> list[tuple[str, str]]:
    """Return (open, close) token pairs for token2json format."""
    return [(f"<s_{tok[1:-1]}>", f"</s_{tok[1:-1]}>") for tok in FIELD_TOKENS]


# ── Processor ─────────────────────────────────────────────────────────────────


def register_field_tokens(
    processor: DonutProcessor, token2json_format: bool = False
) -> DonutProcessor:
    """Register all task + field tokens onto an existing DonutProcessor.

    Mutates `processor` in place (and returns it) so a single processor — the one
    load_model() already built — is the only source of truth for the vocabulary.
    """
    if token2json_format:
        extra = [t for pair in _t2j_pairs() for t in pair] + [MISSING_TOKEN]
    else:
        extra = list(FIELD_TOKENS)
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [TASK_TOKEN] + extra}
    )
    return processor


# ── Label encoding ────────────────────────────────────────────────────────────


def format_label(
    fields: list[dict],
    include_missing: bool = False,
    token2json_format: bool = False,
) -> str:
    """
    Convert annotation fields to a token-wrapped string.

    Derives the field name from the last segment of field_name (after '/'),
    keeping only the first occurrence when duplicates appear.

    token2json_format=False (default — legacy symmetric):
        present : "<E-mail> foo@bar.com <E-mail>"
        missing : "<E-mail><E-mail>"   (only when include_missing=True)

    token2json_format=True (XML-like, parseable with processor.token2json()):
        present : "<s_E-mail>foo@bar.com</s_E-mail>"
        missing : "<s_E-mail><missing></s_E-mail>"

    token2json_format always emits every field in FIELD_TOKENS order — present
    or not — so the model learns a fixed-length, fixed-order template with an
    explicit "absent" signal. `include_missing` only affects the legacy branch.
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
    if token2json_format:
        for open_tok, close_tok in _t2j_pairs():
            leaf = open_tok[3:-1]  # "<s_E-mail>" → "E-mail"
            value = values.get(leaf, "")
            parts.append(f"{open_tok}{value or MISSING_TOKEN}{close_tok}")
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


# ── Label decoding ────────────────────────────────────────────────────────────


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


def _parse_legacy(decoded: str) -> dict[str, str]:
    """
    Parse legacy symmetric-token format into a {field: value} dict.

    The model produces something like:
        <s_donut><E-mail> a@b.com <E-mail><valor_da_nota> 100 <valor_da_nota>

    Lenient by design: a field's value runs from just after its opening token
    to the next field token (including a literal closing duplicate, if the
    model emitted one) or EOS — whichever comes first — rather than requiring
    the literal duplicate closing token. The model emits the closing duplicate
    as a second occurrence of the *same* token id as the opening one, with no
    structural cue distinguishing open from close; when it skips re-emitting
    it, this still recovers the value instead of dropping the whole field.
    """
    boundary = "|".join(re.escape(t) for t in FIELD_TOKENS) + r"|</s>"
    result: dict[str, str] = {}
    for token in FIELD_TOKENS:
        leaf = token[1:-1]  # "<E-mail>" → "E-mail"
        start = re.search(re.escape(token), decoded)
        if not start:
            continue
        rest = decoded[start.end() :]
        end = re.search(boundary, rest)
        value = (rest[: end.start()] if end else rest).strip()
        if value:
            result[leaf] = value
    return result


def parse_prediction(
    decoded: str, token2json_format: bool, processor: DonutProcessor
) -> dict[str, str]:
    """Decode a raw model output string into a {field_name: value} dict."""
    if token2json_format:
        return _parse_token2json(decoded, processor)
    return _parse_legacy(decoded)


# ── Data loading ──────────────────────────────────────────────────────────────


def load_samples(split_json: Path) -> list[dict]:
    split_json = Path(split_json)
    base = split_json.parent
    with open(split_json) as f:
        records = json.load(f)
    return [{"image": base / r["image"], "fields": r["fields"]} for r in records]


# ── Dataset ───────────────────────────────────────────────────────────────────


class DonutDataset(Dataset):
    """
    Args:
        samples:    list of {"image": Path, "fields": [...]} dicts from load_samples()
        processor:  DonutProcessor from build_processor() — must have special tokens registered
        max_length: maximum decoder sequence length (default 128)

    Each item returns:
        pixel_values  (3, H, W) float tensor
        labels        (max_length,) int64 tensor, padding replaced with -100
        target_text   the raw label string, useful for notebook inspection
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
