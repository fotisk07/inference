"""
Dataset inspection and validation for donut fine-tuning.

Run:
    cd donut_train && uv run python inspect_dataset.py
    cd donut_train && uv run python inspect_dataset.py --n 10
    cd donut_train && uv run python inspect_dataset.py --idx 0 3 7
    cd donut_train && uv run python inspect_dataset.py --token2json_format
"""

import argparse
import unicodedata
from pathlib import Path

from dataset import (
    ALL_SPECIAL_TOKENS,
    ALL_SPECIAL_TOKENS_T2J,
    FIELD_TOKENS,
    FIELD_TOKENS_T2J,
    DonutDataset,
    build_processor,
    load_samples,
)
from PIL import Image

_HERE = Path(__file__).parent
MODEL_NAME = "naver-clova-ix/donut-base"
IMAGES_DIR = str(_HERE / "../test_data/images/train")
ANNOTATIONS_DIR = str(_HERE / "../test_data/new_cardxie_annotations/train")
MAX_LENGTH = 128


def _has_garbage(text: str) -> bool:
    for ch in text:
        cat = unicodedata.category(ch)
        # Cc=control, Cs=surrogates, Co=private-use, Cn=unassigned, Cf=format (zero-width space etc.)
        if cat in ("Cc", "Cs", "Co", "Cn", "Cf") and ch not in ("\t", "\n", "\r"):
            return True
    return False


def _show_sample(idx: int, sample: dict, ds: DonutDataset, processor) -> None:
    item = ds[idx]
    target_text = item["target_text"]
    labels = item["labels"]

    real_tokens = int((labels != -100).sum())

    try:
        img = Image.open(sample["image"])
        size_str = f"{img.width}×{img.height}"
    except Exception as e:
        size_str = f"ERROR: {e}"

    tokenized = processor.tokenizer(
        target_text, add_special_tokens=False, return_tensors="pt"
    )
    ids = tokenized.input_ids.squeeze(0).tolist()
    pieces = processor.tokenizer.convert_ids_to_tokens(ids)

    trunc_warn = " ⚠ TRUNCATED" if real_tokens == ds.max_length else ""

    print(
        f"\n  ┌─ Sample {idx}  {Path(sample['image']).name}  [{size_str}]{trunc_warn}"
    )
    print(f"  │  target : {target_text!r}")
    print(f"  │  tokens : {real_tokens}/{ds.max_length}")
    print(f"  │  ids    : {ids[:real_tokens]}")
    print(f"  └─ pieces : {pieces[:real_tokens]}")


def _check_single_piece(processor, token2json_format: bool) -> list[str]:
    tokens = ALL_SPECIAL_TOKENS_T2J if token2json_format else ALL_SPECIAL_TOKENS
    bad = []
    for tok in tokens:
        ids = processor.tokenizer(tok, add_special_tokens=False).input_ids
        if len(ids) != 1:
            bad.append(f"{tok!r} splits into {len(ids)} pieces: {ids}")
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and validate a DonutDataset")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--images_dir", default=IMAGES_DIR)
    parser.add_argument("--annotations_dir", default=ANNOTATIONS_DIR)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument(
        "--n", type=int, default=5, help="Number of samples to show details for"
    )
    parser.add_argument(
        "--idx", type=int, nargs="+", help="Specific sample indices to show"
    )
    parser.add_argument(
        "--token2json_format",
        action="store_true",
        help="Inspect with token2json-compatible encoding (FIELD_TOKENS_T2J)",
    )
    args = parser.parse_args()

    print(f"\n{'═' * 64}")
    print("  Donut dataset inspection")
    print(f"  images     : {args.images_dir}")
    print(f"  annotations: {args.annotations_dir}")
    print(f"  max_length : {args.max_length}")
    print(
        f"  format     : {'token2json (<s_field>…</s_field>)' if args.token2json_format else 'legacy (<field> … <field>)'}"
    )
    print(f"{'═' * 64}")

    print("\nLoading processor (this may download weights the first time)...")
    processor = build_processor(args.model, args.token2json_format)
    samples = load_samples(Path(args.images_dir), Path(args.annotations_dir))

    if not samples:
        print("ERROR: no samples found — check images_dir and annotations_dir")
        return

    print(f"Found {len(samples)} samples\n")
    ds = DonutDataset(
        samples, processor, args.max_length, token2json_format=args.token2json_format
    )

    # ── Per-sample detail ──────────────────────────────────────────────────────
    show_indices = args.idx if args.idx else list(range(min(args.n, len(samples))))
    print(f"{'─' * 64}")
    print(f"  Per-sample breakdown ({len(show_indices)} shown)")

    for i in show_indices:
        if i >= len(samples):
            print(
                f"  WARNING: index {i} out of range (dataset has {len(samples)} samples)"
            )
            continue
        _show_sample(i, samples[i], ds, processor)

    # ── Full-dataset validation ────────────────────────────────────────────────
    print(f"\n{'─' * 64}")
    print("  Validation (full dataset)")
    print(f"{'─' * 64}")

    issues: list[str] = []
    seq_lengths: list[int] = []
    field_counts: dict[str, int] = {}

    # Special-token single-piece check (once, not per sample)
    bad_pieces = _check_single_piece(processor, args.token2json_format)
    if bad_pieces:
        for msg in bad_pieces:
            issues.append(f"[SINGLE PIECE] {msg}")
    else:
        n_tok = len(
            ALL_SPECIAL_TOKENS_T2J if args.token2json_format else ALL_SPECIAL_TOKENS
        )
        print(f"  [SINGLE PIECE] all {n_tok} special tokens → 1 piece each ✓")

    for i, sample in enumerate(samples):
        name = Path(sample["image"]).name

        # IMAGE: can it be opened?
        try:
            img = Image.open(sample["image"])
            img.load()
        except Exception as e:
            issues.append(f"[IMAGE] sample {i} ({name}): {e}")

        # EMPTY LABEL: at least one field has a value
        fields = sample["fields"]
        if not any(f.get("annotator_text", "").strip() for f in fields):
            issues.append(f"[EMPTY LABEL] sample {i} ({name}): all fields are empty")

        # GARBAGE: control/format chars in field values
        for f in fields:
            val = f.get("annotator_text", "")
            if _has_garbage(val):
                issues.append(
                    f"[GARBAGE] sample {i} ({name}) field {f['field_name']!r}: {val!r}"
                )

        # TRUNCATED: real content fits in max_length
        full_ids = processor.tokenizer(
            ds[i]["target_text"], add_special_tokens=False
        ).input_ids
        real_tokens = int((ds[i]["labels"] != -100).sum())
        seq_lengths.append(real_tokens)
        if len(full_ids) > args.max_length:
            issues.append(
                f"[TRUNCATED] sample {i} ({name}): {len(full_ids)} tokens"
                f" > max_length={args.max_length} — content is cut off"
            )

        # field frequency tally
        for f in fields:
            if f.get("annotator_text", "").strip():
                leaf = f["field_name"].split("/")[-1]
                field_counts[leaf] = field_counts.get(leaf, 0) + 1

    if issues:
        print(f"\n  ⚠  {len(issues)} issue(s) found:\n")
        for iss in issues:
            print(f"    {iss}")
    else:
        print("  All checks passed ✓")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 64}")
    print("  Summary")
    print(f"{'─' * 64}")

    if seq_lengths:
        print(
            f"  Sequence lengths (real tokens):"
            f"  min={min(seq_lengths)}"
            f"  max={max(seq_lengths)}"
            f"  mean={sum(seq_lengths) / len(seq_lengths):.1f}"
            f"  max_length={args.max_length}"
        )

    print(f"\n  Field frequency (non-empty occurrences across {len(samples)} samples):")
    field_names = (
        [t[1:-1] for t in FIELD_TOKENS]
        if not args.token2json_format
        else [t[0][3:-1] for t in FIELD_TOKENS_T2J]
    )
    for leaf in field_names:
        count = field_counts.get(leaf, 0)
        bar = "█" * count
        print(f"    {leaf:30s}  {count:3d}  {bar}")

    print(f"\n{'═' * 64}\n")


if __name__ == "__main__":
    main()
