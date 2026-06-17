"""Donut fine-tuning training script."""

import json
import random
import time
from pathlib import Path

import mlflow
import torch
from tqdm import tqdm
from dataset import DonutDataset, build_processor, load_samples
from donut import load_model
from predict import predict_and_score
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


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

    # When set, after training run a field-level eval on the validation split and
    # write a one-line ablation summary (config + F1 + speed) to this JSON path.
    # This is what the scripts/exp_*.sh sweeps collect for notebooks/ablation.ipynb.
    ablation_out: str | None = None

    # Set to an experiment name to enable MLflow logging, None to disable.
    mlflow_experiment: str | None = None
    # Name shown in the MLflow UI for this run; auto-generated from lr+bs if None.
    run_name: str | None = None

    # When True (default): encodes fields as <s_field>value</s_field>, always
    # emitting every field (missing ones get a <missing> placeholder) — output
    # parseable with processor.token2json(seq) after stripping task/pad/EOS
    # tokens. When False: legacy symmetric format <field> value <field>.
    token2json_format: bool = True

    # Acceleration backend passed to donut.load_model(). "eager" disables all
    # patches except mask caching; "sdpa" adds PyTorch SDPA; "fa" requires CUDA + flash-attn.
    backend: str = "sdpa"

    grad_clip: float = 1.0
    weight_decay: float = 0.01
    # Set to fix random.shuffle order and torch ops for reproducible val splits.
    seed: int | None = None

    # When True: use the tiny random model from donut.synthetic on a small data
    # subset. Runs on CPU in seconds with no HF downloads. Zero exit = pipeline OK.
    smoke: bool = False
    smoke_n_samples: int = 4
    smoke_epochs: int = 5

    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


# ── Model setup ───────────────────────────────────────────────────────────────
def build_model(cfg: "Config", processor):
    if cfg.smoke:
        from donut.synthetic import make_tiny_model

        model = make_tiny_model()
    else:
        model, _ = load_model(
            model_id=cfg.model_name, device=cfg.device, backend=cfg.backend
        )
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    # naver-standard: pad as decoder start, task token lives in labels
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.pad_token_id
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> float | None:
    if len(loader) == 0:
        return None
    model.eval()
    model_dtype = next(model.parameters()).dtype
    total_loss = 0.0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device=device, dtype=model_dtype)
        labels = batch["labels"].to(device)
        total_loss += model(pixel_values=pixel_values, labels=labels).loss.item()
    return total_loss / len(loader)


# ── Training loop ─────────────────────────────────────────────────────────────
def train(config: Config) -> None:
    if config.smoke:
        config = config.model_copy(
            update=dict(
                device="cpu",
                image_size=(64, 64),
                batch_size=1,
                num_workers=0,
                warmup_steps=0,
                max_epochs=config.smoke_epochs,
                mlflow_experiment=None,
            )
        )

    print(f"\n{'=' * 60}")
    print("  Donut fine-tuning" + ("  [SMOKE TEST]" if config.smoke else ""))
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

    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
    random.shuffle(samples)
    if config.smoke:
        samples = samples[: config.smoke_n_samples]
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
    pin_memory = "cuda" in config.device
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    # --- model ---
    if config.smoke:
        print("Loading tiny model (donut.synthetic, CPU, no downloads) ...")
    else:
        print(f"Loading model from {config.model_name} (backend={config.backend}) ...")
    model = build_model(config, processor)
    model_dtype = next(model.parameters()).dtype

    # --- optimizer + scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    total_steps = len(train_loader) * config.max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"Total steps: {total_steps}  warmup: {config.warmup_steps}\n")

    if not config.smoke:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(config.max_epochs):
        # --- train epoch ---
        model.train()
        epoch_loss = 0.0
        docs_trained = 0
        epoch_start = time.time()

        batch_bar = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{config.max_epochs}",
            leave=False,
        )
        for batch in batch_bar:
            pixel_values = batch["pixel_values"].to(
                device=config.device, dtype=model_dtype
            )
            labels = batch["labels"].to(config.device)

            loss = model(pixel_values=pixel_values, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            docs_trained += pixel_values.shape[0]
            global_step += 1
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
            if config.mlflow_experiment:
                mlflow.log_metric("train_loss_step", loss.item(), step=global_step)

        epoch_secs = time.time() - epoch_start
        train_loss = epoch_loss / len(train_loader)
        docs_per_sec = docs_trained / epoch_secs

        # --- validation ---
        val_loss = evaluate(model, val_loader, config.device)
        model.train()

        val_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
        print(
            f"Epoch {epoch + 1:3d}/{config.max_epochs}"
            f"  train={train_loss:.4f}  val={val_str}"
            f"  │  {docs_per_sec:.1f} docs/s  {epoch_secs:.0f}s"
        )
        if config.mlflow_experiment:
            metrics = {"train_loss": train_loss, "docs_per_sec": docs_per_sec}
            if val_loss is not None:
                metrics["val_loss"] = val_loss
            mlflow.log_metrics(metrics, step=epoch + 1)

        # --- checkpoint ---
        if not config.smoke and (epoch + 1) % config.save_every_n_epochs == 0:
            ckpt_path = (
                Path(config.output_dir) / f"epoch_{epoch + 1:03d}_val{val_loss:.4f}.pt"
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    # metadata needed by predict.py to reconstruct the model
                    "model_name": config.model_name,
                    "token2json_format": config.token2json_format,
                    "image_size": config.image_size,
                    "max_length": config.max_length,
                },
                ckpt_path,
            )
            print(f"           saved → {ckpt_path}")

    print("\nTraining complete.")

    # --- ablation summary: field-level F1 on the val split (final-epoch weights) ---
    if config.ablation_out and val_samples:
        print("\nScoring validation split for ablation summary ...")
        quality = predict_and_score(
            model,
            processor,
            val_samples,
            token2json_format=config.token2json_format,
            device=config.device,
            max_new_tokens=config.max_length,
        )
        summary = {
            "image_size": list(config.image_size),
            "batch_size": config.batch_size,
            "lr": config.lr,
            "max_epochs": config.max_epochs,
            "backend": config.backend,
            "model_name": config.model_name,
            "n_train": len(train_samples),
            "final_val_loss": val_loss,
            "docs_per_sec": docs_per_sec,
            "f1_strict": quality["strict"]["f1"],
            "f1_soft": quality["soft"]["f1"],
            "quality": quality,
            "config": config.model_dump(),
        }
        out_path = Path(config.ablation_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Ablation summary → {out_path}  (f1_strict={summary['f1_strict']})")
        if config.mlflow_experiment:
            for k in ("f1_strict", "f1_soft"):
                if summary[k] is not None:
                    mlflow.log_metric(k, summary[k])

    if config.mlflow_experiment:
        mlflow.end_run()


if __name__ == "__main__":
    train(Config())
