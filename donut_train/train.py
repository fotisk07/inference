import random
from dataclasses import dataclass
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import DonutProcessor

from dataset import DonutDataset, load_local_samples
from label_formatter import LabelFormatter
from model_module import DonutModule


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    model_name: str = "naver-clova-ix/donut-base"

    images_dir: str = "data/images"
    annotations_dir: str = "data/annotations"
    val_split: float = 0.2

    task_start_token: str = "<s_donut>"

    max_target_length: int = 512
    image_size: tuple[int, int] = (1280, 960)

    batch_size: int = 4
    num_workers: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 300
    max_epochs: int = 30


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DonutDataModule(L.LightningDataModule):
    def __init__(self, config: Config, processor: DonutProcessor, label_formatter: LabelFormatter):
        super().__init__()
        self.config = config
        self.processor = processor
        self.label_formatter = label_formatter

    def setup(self, stage=None):
        samples = load_local_samples(
            images_dir=Path(self.config.images_dir),
            annotations_dir=Path(self.config.annotations_dir),
        )
        random.shuffle(samples)

        split = int(len(samples) * (1 - self.config.val_split))
        train_samples = samples[:split]
        val_samples = samples[split:]

        self.train_dataset = DonutDataset(
            data=train_samples,
            processor=self.processor,
            label_formatter=self.label_formatter,
            task_start_token=self.config.task_start_token,
            max_target_length=self.config.max_target_length,
        )
        self.val_dataset = DonutDataset(
            data=val_samples,
            processor=self.processor,
            label_formatter=self.label_formatter,
            task_start_token=self.config.task_start_token,
            max_target_length=self.config.max_target_length,
        )

        print(f"Dataset: {len(train_samples)} train, {len(val_samples)} val samples")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    config = Config()

    label_formatter = LabelFormatter()
    all_tokens = [config.task_start_token] + LabelFormatter.get_all_tokens()

    processor = DonutProcessor.from_pretrained(config.model_name)
    processor.feature_extractor.size = {"height": config.image_size[0], "width": config.image_size[1]}

    model = DonutModule(
        model_name=config.model_name,
        processor=processor,
        additional_tokens=all_tokens,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
    )
    data_module = DonutDataModule(config, processor, label_formatter)

    # ── Overfit-one-batch sanity check ──────────────────────────────────────
    # Uncomment this block to verify the training loop is working correctly.
    # The loss should drop close to zero within ~20 epochs on a single batch.
    # If it doesn't, something is wrong with the data pipeline or loss computation.
    #
    # trainer = L.Trainer(overfit_batches=1, max_epochs=20, accelerator="auto")
    # trainer.fit(model, datamodule=data_module)
    # return
    # ────────────────────────────────────────────────────────────────────────

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        callbacks=[
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3, filename="{epoch}-{val_loss:.4f}"),
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
