import random
import sys
from pathlib import Path

from PIL import Image


def load_pool(
    pool_size: int,
    dataset: str | None,
    dataset_split: str,
    image_column: str,
    image_dir: str | None,
) -> list:
    """Return a lazy pool of Path objects (local) or (dataset, column, idx) tuples (HF)."""
    if image_dir:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        paths = sorted(
            p for p in Path(image_dir).iterdir() if p.suffix.lower() in extensions
        )
        if not paths:
            sys.exit(f"No images found in {image_dir}")
        pool = paths[:pool_size]
    else:
        from datasets import load_dataset

        ds = load_dataset(dataset, split=dataset_split)
        n = min(pool_size, len(ds))
        pool = [(ds, image_column, i) for i in range(n)]
    if not pool:
        sys.exit("Pool is empty — check --dataset / --image-dir / --pool")
    return pool


def _load_entry(entry) -> Image.Image:
    if isinstance(entry, Path):
        return Image.open(entry).convert("RGB")
    ds, col, i = entry
    return ds[i][col].convert("RGB")


def sample_batch(pool: list, batch_size: int) -> list[Image.Image]:
    entries = [pool[random.randrange(len(pool))] for _ in range(batch_size)]
    return [_load_entry(e) for e in entries]
