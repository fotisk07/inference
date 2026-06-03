import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from transformers import DonutProcessor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_local_samples(images_dir: Path, annotations_dir: Path) -> list[dict]:
    """
    Pairs each image in images_dir with its matching JSON annotation file.
    Skips images that have no corresponding annotation.
    Returns a list of {"image": Path, "fields": [...]} dicts (images are not loaded yet).
    """
    samples = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        ann_path = annotations_dir / (img_path.stem + ".json")
        if not ann_path.exists():
            continue
        with open(ann_path) as f:
            annotation = json.load(f)
        samples.append({"image": img_path, "fields": annotation["fields"]})
    return samples


class DonutDataset(Dataset):
    def __init__(
        self,
        data: list[dict],
        processor: DonutProcessor,
        label_formatter,
        task_start_token: str,
        max_target_length: int = 512,
    ):
        self.data = data
        self.processor = processor
        self.label_formatter = label_formatter
        self.task_start_token = task_start_token
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        img = sample["image"]
        image = (Image.open(img) if isinstance(img, (str, Path)) else img).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        target_text = self.task_start_token + self.label_formatter.format(sample)
        tokenized = self.processor.tokenizer(
            target_text,
            add_special_tokens=False,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.squeeze(0).clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}
