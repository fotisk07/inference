"""Donut fine-tuning training script."""

import random
import time
from pathlib import Path

import mlflow
import torch
from dataset import DonutDataset, build_processor, load_samples
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, get_linear_schedule_with_warmup


# ── Config ────────────────────────────────────────────────────────────────────
class Config(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)

    model_name: str = "naver-clova-ix/donut-base"

    data_json: str = "../test_data/train.json"
    val_split: float = 0.2

    # (height, width) — donut-base pretrain default; reduce to save GPU memory
    image_size: tuple[int, int] = (1280, 960)

    max_length: int = 128
    batch_size: int = 4
    num_workers: int = 4

    lr: float = 3e-4
    warmup_steps: int = 100
    max_epochs: int = 30

    output_dir: str = "checkpoints"
    save_every_n_epochs: int = 1

    # Set to an experiment name to enable MLflow logging, None to disable.
    mlflow_experiment: str | None = None
    # Name shown in the MLflow UI for this run; auto-generated from lr+bs if None.
    run_name: str | None = None

    # When True: encodes fields as <s_field>value</s_field> — output parseable
    # with processor.token2json(seq) after stripping task/pad/EOS tokens.
    # When False (default): legacy symmetric format <field> value <field>.
    token2json_format: bool = False

    device: str = Field(
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
    print("  Donut fine-tuning")
    print(
        f"  device={config.device}  lr={config.lr}  batch={config.batch_size}"
        f"  epochs={config.max_epochs}  image={config.image_size[0]}×{config.image_size[1]}"
    )
    print(f"{'=' * 60}\n")

    # --- mlflow ---
    if config.mlflow_experiment:
        mlflow.set_experiment(config.mlflow_experiment)
        run_name = config.run_name or f"lr{config.lr}-bs{config.batch_size}"
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(config.model_dump())

    # --- data ---
    processor = build_processor(config.model_name, config.token2json_format)
    processor.image_processor.size = {
        "height": config.image_size[0],
        "width": config.image_size[1],
    }
    samples = load_samples(Path(config.data_json))

    random.shuffle(samples)
    split = max(1, int(len(samples) * (1 - config.val_split)))
    train_samples, val_samples = samples[:split], samples[split:]
    print(f"Dataset: {len(train_samples)} train, {len(val_samples)} val")

    train_ds = DonutDataset(
        train_samples,
        processor,
        config.max_length,
        token2json_format=config.token2json_format,
    )
    val_ds = DonutDataset(
        val_samples,
        processor,
        config.max_length,
        token2json_format=config.token2json_format,
    )
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
        docs_trained = 0
        epoch_start = time.time()

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(config.device)
            labels = batch["labels"].to(config.device)

            loss = model(pixel_values=pixel_values, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            docs_trained += pixel_values.shape[0]
            global_step += 1
            if config.mlflow_experiment:
                mlflow.log_metric("train_loss_step", loss.item(), step=global_step)

        epoch_secs = time.time() - epoch_start
        train_loss = epoch_loss / len(train_loader)
        docs_per_sec = docs_trained / epoch_secs

        # --- validation ---
        val_loss = evaluate(model, val_loader, config.device)
        model.train()

        print(
            f"Epoch {epoch + 1:3d}/{config.max_epochs}"
            f"  train={train_loss:.4f}  val={val_loss:.4f}"
            f"  │  {docs_per_sec:.1f} docs/s  {epoch_secs:.0f}s"
        )
        if config.mlflow_experiment:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "docs_per_sec": docs_per_sec,
                },
                step=epoch + 1,
            )

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
    if config.mlflow_experiment:
        mlflow.end_run()


if __name__ == "__main__":
    train(Config())
