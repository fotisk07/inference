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
) -> list[Image.Image]:
    """Load an image pool from a local directory or HuggingFace dataset."""
    if image_dir:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        paths = sorted(
            p for p in Path(image_dir).iterdir() if p.suffix.lower() in extensions
        )
        if not paths:
            sys.exit(f"No images found in {image_dir}")
        images = [Image.open(p).convert("RGB") for p in paths[:pool_size]]
    else:
        from datasets import load_dataset

        ds = load_dataset(dataset, split=dataset_split)
        n = min(pool_size, len(ds))
        images = [ds[i][image_column].convert("RGB") for i in range(n)]
    if not images:
        sys.exit("Pool is empty — check --dataset / --image-dir / --pool")
    return images


def sample_batch(pool: list[Image.Image], batch_size: int) -> list[Image.Image]:
    return [pool[random.randrange(len(pool))] for _ in range(batch_size)]
