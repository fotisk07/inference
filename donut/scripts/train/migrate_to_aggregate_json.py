"""
Migrate per-image JSON annotations to a single aggregate JSON file.

Input (current format):
    images_dir/       doc001.jpg, doc002.jpg, ...
    annotations_dir/  doc001.json, doc002.json, ...
        each file: {"fields": [{"field_name": "...", "annotator_text": "..."}, ...]}

Output (aggregate format):
    output.json
        [{"image": "<path relative to output.json>", "fields": [...]}, ...]

Usage:
    uv run python scripts/train/migrate_to_aggregate_json.py \\
        --images_dir /data/images/train \\
        --annotations_dir /data/annotations/train \\
        --output /data/train.json

    # dry run — print summary without writing
    uv run python scripts/train/migrate_to_aggregate_json.py ... --dry_run
"""

import argparse
import json
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def migrate(
    images_dir: Path, annotations_dir: Path, output: Path, dry_run: bool
) -> None:
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)
    output = Path(output)

    records = []
    skipped = []

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        ann_path = annotations_dir / (img_path.stem + ".json")
        if not ann_path.exists():
            skipped.append(img_path.name)
            continue

        with open(ann_path) as f:
            annotation = json.load(f)

        # store image path relative to the output JSON so the tree is relocatable
        try:
            rel_image = img_path.resolve().relative_to(output.parent.resolve())
        except ValueError:
            # can't make relative — fall back to absolute
            rel_image = img_path.resolve()

        records.append({"image": str(rel_image), "fields": annotation["fields"]})

    print(f"  Found   : {len(records)} paired samples")
    if skipped:
        print(f"  Skipped : {len(skipped)} images with no annotation")
        for name in skipped:
            print(f"            {name}")
    print(f"  Output  : {output}")

    if dry_run:
        print("\n  [dry run] — not writing")
        if records:
            print("\n  First record preview:")
            print(json.dumps(records[0], indent=2, ensure_ascii=False))
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\n  Written {len(records)} records → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate per-image JSONs to aggregate JSON"
    )
    parser.add_argument(
        "--images_dir", required=True, help="Directory containing images"
    )
    parser.add_argument(
        "--annotations_dir",
        required=True,
        help="Directory containing per-image JSON annotations",
    )
    parser.add_argument(
        "--output", required=True, help="Output aggregate JSON file path"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Preview without writing"
    )
    args = parser.parse_args()

    print("\nMigrating dataset")
    print(f"  images_dir      : {args.images_dir}")
    print(f"  annotations_dir : {args.annotations_dir}")

    migrate(
        images_dir=Path(args.images_dir),
        annotations_dir=Path(args.annotations_dir),
        output=Path(args.output),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
