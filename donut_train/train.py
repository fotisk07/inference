from dataclasses import dataclass, field

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import DonutProcessor

from dataset import DonutDataset
from model_module import DonutModule


# ---------------------------------------------------------------------------
# Replace this with your real LabelFormatter import, e.g.:
#   from label_formatter import LabelFormatter
# ---------------------------------------------------------------------------
class LabelFormatter:
    """Stub — replace with your actual implementation."""

    def format(self, sample: dict) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    model_name: str = "naver-clova-ix/donut-base"
    dataset_name: str = "naver-clova-ix/cord-v2"  # swap for your own HF dataset

    # Token added at the start of every target sequence; also used as decoder_start_token
    task_start_token: str = "<s_donut>"
    # Any extra domain tokens the LabelFormatter uses (e.g. <champ>)
    additional_tokens: list[str] = field(default_factory=lambda: ["<champ>"])

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
        from datasets import load_dataset

        raw = load_dataset(self.config.dataset_name)
        all_tokens = [self.config.task_start_token] + self.config.additional_tokens

        self.train_dataset = DonutDataset(
            data=raw["train"],
            processor=self.processor,
            label_formatter=self.label_formatter,
            task_start_token=self.config.task_start_token,
            max_target_length=self.config.max_target_length,
        )
        self.val_dataset = DonutDataset(
            data=raw["validation"],
            processor=self.processor,
            label_formatter=self.label_formatter,
            task_start_token=self.config.task_start_token,
            max_target_length=self.config.max_target_length,
        )

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

    processor = DonutProcessor.from_pretrained(config.model_name)
    processor.feature_extractor.size = {"height": config.image_size[0], "width": config.image_size[1]}

    all_tokens = [config.task_start_token] + config.additional_tokens
    label_formatter = LabelFormatter()  # swap with your real class

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
