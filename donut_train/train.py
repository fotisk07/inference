"""
Donut fine-tuning training script.

Configure via the Config dataclass at the top, then run:
    cd donut_train && uv run python train.py

To add MLflow, gradient clipping, early stopping, etc., look for
the "# extend:" comments — those are the exact spots to add them.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, get_linear_schedule_with_warmup

from dataset import DonutDataset, build_processor, load_samples


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    model_name: str = "naver-clova-ix/donut-base"

    images_dir: str = "../test_data/images/train"
    annotations_dir: str = "../test_data/new_cardxie_annotations/train"
    val_split: float = 0.2

    max_length: int = 128
    batch_size: int = 4
    num_workers: int = 4

    lr: float = 3e-4
    warmup_steps: int = 100
    max_epochs: int = 30

    output_dir: str = "checkpoints"
    save_every_n_epochs: int = 1

    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


# ── Model setup ───────────────────────────────────────────────────────────────
def build_model(processor, model_name: str) -> VisionEncoderDecoderModel:
    """Load pretrained model and configure it for fine-tuning."""
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    # naver-standard: pad as decoder start, task token lives in labels
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.pad_token_id
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: VisionEncoderDecoderModel, loader: DataLoader, device: str
) -> float:
    model.eval()
    total_loss = 0.0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        total_loss += model(pixel_values=pixel_values, labels=labels).loss.item()
    return total_loss / len(loader)


# ── Training loop ─────────────────────────────────────────────────────────────
def train(config: Config) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Donut fine-tuning")
    print(
        f"  device={config.device}  lr={config.lr}  batch={config.batch_size}  epochs={config.max_epochs}"
    )
    print(f"{'=' * 60}\n")

    # --- data ---
    processor = build_processor(config.model_name)
    samples = load_samples(Path(config.images_dir), Path(config.annotations_dir))
    if not samples:
        raise ValueError(f"No samples found in {config.images_dir}")

    random.shuffle(samples)
    split = max(1, int(len(samples) * (1 - config.val_split)))
    train_samples, val_samples = samples[:split], samples[split:]
    print(f"Dataset: {len(train_samples)} train, {len(val_samples)} val")

    train_ds = DonutDataset(train_samples, processor, config.max_length)
    val_ds = DonutDataset(val_samples, processor, config.max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # --- model ---
    print(f"Loading model from {config.model_name} ...")
    model = build_model(processor, config.model_name).to(config.device)

    # --- optimizer + scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    total_steps = len(train_loader) * config.max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"Total steps: {total_steps}  warmup: {config.warmup_steps}\n")

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(config.max_epochs):
        # --- train epoch ---
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(config.device)
            labels = batch["labels"].to(config.device)

            loss = model(pixel_values=pixel_values, labels=labels).loss
            loss.backward()
            # extend: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1
            # extend: mlflow.log_metric("train_loss_step", loss.item(), step=global_step)

        train_loss = epoch_loss / len(train_loader)

        # --- validation ---
        val_loss = evaluate(model, val_loader, config.device)
        model.train()

        print(
            f"Epoch {epoch + 1:3d}/{config.max_epochs}  train={train_loss:.4f}  val={val_loss:.4f}"
        )
        # extend: mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        # --- checkpoint ---
        if (epoch + 1) % config.save_every_n_epochs == 0:
            ckpt_path = (
                Path(config.output_dir) / f"epoch_{epoch + 1:03d}_val{val_loss:.4f}.pt"
            )
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                ckpt_path,
            )
            print(f"           saved → {ckpt_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    train(Config())
