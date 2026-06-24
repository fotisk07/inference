"""Donut fine-tuning training script."""

import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mlflow
import torch
import typer
from tqdm import tqdm
from dataset import TASK_TOKEN, DonutDataset, load_samples, register_field_tokens
from donut import load_model
from torch.utils.data import DataLoader
from transformers import DonutProcessor, get_linear_schedule_with_warmup


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    """Typed bundle of training settings, built from the CLI by `main` below.

    image_size is (height, width). token2json_format=True encodes fields as
    <s_field>value</s_field> (parseable via processor.token2json); False is the
    legacy symmetric format. backend/precision are passed through to the donut
    accel path. See `main` for per-flag defaults and help.
    """

    model_name: str
    data_json: str
    val_split: float
    image_size: tuple[int, int]
    max_length: int
    batch_size: int
    num_workers: int
    lr: float
    warmup_steps: int
    max_epochs: int
    output_dir: str
    mlflow_experiment: str | None
    run_name: str | None
    token2json_format: bool
    backend: str
    precision: str
    grad_clip: float
    weight_decay: float
    seed: int | None
    smoke: bool
    smoke_n_samples: int
    smoke_epochs: int
    device: str


# ── Precision ─────────────────────────────────────────────────────────────────
def autocast(device: str, precision: str):
    """bf16 autocast on CUDA when precision=="bf16"; an inert no-op otherwise."""
    enabled = device.startswith("cuda") and precision == "bf16"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return torch.autocast(
        device_type=device_type, dtype=torch.bfloat16, enabled=enabled
    )


# ── Model setup ───────────────────────────────────────────────────────────────
def build_model(cfg: "Config"):
    """Return (model, processor) with field tokens registered and embeddings resized.

    Weights load in fp32 (master weights for stable fine-tuning); bf16 compute is
    applied at the forward via autocast(). The accel backend stays active either way.
    """
    if cfg.smoke:
        from donut.synthetic import make_tiny_model

        model = make_tiny_model()
        processor = DonutProcessor.from_pretrained(cfg.model_name)
    else:
        model, processor = load_model(
            model_id=cfg.model_name,
            device=cfg.device,
            dtype=torch.float32,
            backend=cfg.backend,
        )
    register_field_tokens(processor, cfg.token2json_format)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    # Canonical Donut: the task token is the decoder start (auto-prepended to the
    # labels by shift_tokens_right); pad stays pad for loss masking / padding.
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        TASK_TOKEN
    )
    return model, processor


def save_checkpoint(model, processor, out_dir: Path, token2json_format: bool) -> None:
    """Save a portable HF artifact (model + processor) plus a small meta sidecar.

    image_size lives in the processor's image-processor config; token2json_format is
    the one thing save_pretrained doesn't capture, so predict.py reads it back from
    train_meta.json to pick the right output parser.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    (out_dir / "train_meta.json").write_text(
        json.dumps({"token2json_format": token2json_format})
    )


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: str, precision: str
) -> float | None:
    if len(loader) == 0:
        return None
    model.eval()
    total_loss = 0.0
    for batch in tqdm(loader, desc="val", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        with autocast(device, precision):
            total_loss += model(pixel_values=pixel_values, labels=labels).loss.item()
    return total_loss / len(loader)


def _seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker so augmentation/order is reproducible."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)


# ── Training loop ─────────────────────────────────────────────────────────────
def train(config: Config) -> None:
    if config.smoke:
        config = replace(
            config,
            device="cpu",
            image_size=(64, 64),
            batch_size=1,
            num_workers=0,
            warmup_steps=0,
            max_epochs=config.smoke_epochs,
            mlflow_experiment=None,
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
        mlflow.log_params(asdict(config))

    # --- model + processor (single processor is the sole vocab source) ---
    if config.smoke:
        print("Loading tiny model (donut.synthetic, CPU, no downloads) ...")
    else:
        print(f"Loading model from {config.model_name} (backend={config.backend}) ...")
    model, processor = build_model(config)
    processor.image_processor.size = {
        "height": config.image_size[0],
        "width": config.image_size[1],
    }

    # --- data ---
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
    # Deterministic shuffle order when a seed is set (not just the split).
    generator = None
    if config.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=_seed_worker if config.seed is not None else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

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

    global_step = 0
    best_val_loss = float("inf")

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
            pixel_values = batch["pixel_values"].to(config.device)
            labels = batch["labels"].to(config.device)

            with autocast(config.device, config.precision):
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
        val_loss = evaluate(model, val_loader, config.device, config.precision)
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

        # --- checkpoint: keep the best-by-val-loss ---
        if not config.smoke and val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = Path(config.output_dir) / "best"
            save_checkpoint(model, processor, best_dir, config.token2json_format)
            print(f"           saved best (val={val_loss:.4f}) → {best_dir}")

    # --- checkpoint: final weights ---
    if not config.smoke:
        last_dir = Path(config.output_dir) / "last"
        save_checkpoint(model, processor, last_dir, config.token2json_format)
        print(f"Saved last → {last_dir}")

    print("\nTraining complete.")

    if config.mlflow_experiment:
        mlflow.end_run()


# ── CLI ─────────────────────────────────────────────────────────────────────--
app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_name: str = "naver-clova-ix/donut-base",
    data_json: str = "../test_data/train.json",
    val_split: float = 0.2,
    # (height, width) fed to the encoder; lower = faster + less VRAM, less legible.
    image_height: int = 1280,
    image_width: int = 960,
    max_length: int = 128,
    batch_size: int = 4,
    num_workers: int = 4,
    lr: float = 3e-4,
    warmup_steps: int = 100,
    max_epochs: int = 30,
    # Parent of the best/ (lowest val loss) and last/ save_pretrained dirs.
    output_dir: str = "checkpoints",
    # Set an experiment name to enable MLflow logging; run_name defaults to lr+bs.
    mlflow_experiment: str | None = None,
    run_name: str | None = None,
    token2json_format: bool = True,
    # donut accel backend, active in training: eager/sdpa/fa.
    backend: str = "sdpa",
    # On CUDA: "bf16" = fp32 master weights + bf16 autocast compute; "fp32" = all fp32.
    precision: str = "bf16",
    grad_clip: float = 1.0,
    weight_decay: float = 0.01,
    # Set (e.g. 42) for reproducible shuffle order + split.
    seed: int | None = None,
    # Tiny offline model on CPU, few samples — proves the pipeline (CI). No saving.
    smoke: bool = False,
    smoke_n_samples: int = 4,
    smoke_epochs: int = 5,
    # Defaults to cuda when available, else cpu.
    device: str | None = None,
) -> None:
    """Fine-tune Donut for field extraction."""
    train(
        Config(
            model_name=model_name,
            data_json=data_json,
            val_split=val_split,
            image_size=(image_height, image_width),
            max_length=max_length,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            warmup_steps=warmup_steps,
            max_epochs=max_epochs,
            output_dir=output_dir,
            mlflow_experiment=mlflow_experiment,
            run_name=run_name,
            token2json_format=token2json_format,
            backend=backend,
            precision=precision,
            grad_clip=grad_clip,
            weight_decay=weight_decay,
            seed=seed,
            smoke=smoke,
            smoke_n_samples=smoke_n_samples,
            smoke_epochs=smoke_epochs,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
    )


if __name__ == "__main__":
    app()
