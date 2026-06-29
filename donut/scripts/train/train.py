"""Donut fine-tuning — owns the whole training stack: the model build, the per-epoch
loop, evaluation, checkpointing, and the CLI. The reusable model-config helpers it
leans on (autocast, set_donut_shift_tokens, fit_decoder_to_vocab, set_image_size)
live in donut.model / donut.dataset; everything training-specific lives here.
"""

import random
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import mlflow
import torch
import typer
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DonutProcessor, get_linear_schedule_with_warmup

from donut import check_accel, decoder_attn_impl
from donut.constants import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_LENGTH,
    MODEL_ID,
    GLOBAL_OUT_DIR,
)
from donut.dataset import (
    DonutDataset,
    load_samples,
    register_field_tokens,
)
from donut.model import (
    autocast,
    fit_decoder_to_vocab,
    load_model,
    set_donut_shift_tokens,
    set_image_size,
)
from donut.runio import run_meta, save_record

# Repo root (…/inference) so the default --data-json resolves no matter the CWD.
_REPO_ROOT = Path(__file__).resolve().parents[3]


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    """Typed bundle of training settings, built from the CLI by `main` below.

    image_size is (height, width). Labels are encoded token2json-style as
    <s_field>value</s_field> (parseable via processor.token2json). backend/precision
    are passed through to the donut accel path. See `main` for per-flag defaults.
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
    output_dir: Path
    mlflow_experiment: str | None
    run_name: str | None
    backend: str
    precision: str
    grad_clip: float
    weight_decay: float
    seed: int | None
    smoke: bool
    smoke_n_samples: int
    smoke_epochs: int
    device: str


def _seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker so augmentation/order is reproducible."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)


def _sync(device: str) -> None:
    """Make GPU work finish before a timing read; no-op on CPU."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


# ── Model setup ───────────────────────────────────────────────────────────────
def build_model(model_name: str, device: str, backend: str, *, smoke: bool = False):
    """Return (model, processor) ready to fine-tune: field tokens registered,
    decoder embeddings grown to fit, and the shift-token ids set.

    Weights load in fp32 (master weights for stable fine-tuning); bf16 compute is
    applied at the forward via autocast(). In smoke mode the model is the tiny offline
    fixture (no weight download); the processor still comes from `model_name`.
    """
    if smoke:
        from donut.synthetic import make_tiny_model

        model = make_tiny_model()
        processor = DonutProcessor.from_pretrained(model_name)
    else:
        model, processor = load_model(
            model_id=model_name, device=device, dtype=torch.float32, backend=backend
        )
    register_field_tokens(processor)
    fit_decoder_to_vocab(model, processor)
    # Canonical Donut: the task token is the decoder start (auto-prepended to the
    # labels by shift_tokens_right); pad stays pad for loss masking / padding.
    set_donut_shift_tokens(model, processor)
    return model, processor


def save_checkpoint(model, processor, out_dir: Path) -> None:
    """Save a portable HF artifact: the model + the processor. save_pretrained
    captures the weights, the added field tokens, image size, and decoder_start —
    everything predict.py needs to rebuild the model, so there is no sidecar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: str, precision: str
) -> tuple[float | None, float | None]:
    """Return (mean val loss, val compute docs/s).

    Forward-only under no_grad/eval, so per-doc this is faster than a train step
    (no backward, no optimizer). docs/s is measured over the synced compute region.
    """
    if len(loader) == 0:
        return None, None
    model.eval()
    total_loss = 0.0
    docs = 0
    compute = 0.0
    for batch in tqdm(loader, desc="validation", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        _sync(device)
        c0 = time.perf_counter()
        with autocast(device, precision):
            loss = model(pixel_values=pixel_values, labels=labels).loss
        _sync(device)
        compute += time.perf_counter() - c0
        total_loss += loss.item()
        docs += pixel_values.shape[0]
    docs_s = docs / compute if compute > 0 else None
    return total_loss / len(loader), docs_s


# ── Reporting ─────────────────────────────────────────────────────────────────
def docs_per_s(n: int, secs: float) -> float | None:
    """The one throughput formula (see README.md Metrics): docs/s = n / Δt."""
    return n / secs if secs > 0 else None


def _round(x: float | None, n: int) -> float | None:
    """Round for clean records/printing; pass None through."""
    return round(x, n) if x is not None else None


def epoch_record(
    epoch: int,
    *,
    wall: float,
    compute: float,
    docs: int,
    steps: int,
    loss_sum: float,
    val_loss: float | None,
    val_docs_s: float | None,
) -> dict:
    """Per-epoch metrics. The wall splits TWO ways (summing to 100%):
      compute% — synced fwd+bwd+clip+opt (same op set as bench_train_step)
      overhead% — everything else: H2D + scheduler.step + .item()/postfix/mlflow/sync
    `compute% == e2e_docs_s / compute_docs_s`. data% is intentionally NOT split out —
    under prefetch it is fungible with overhead; measure the loader with bench_loader.py.
    """
    overhead = wall - compute
    pct = (lambda t: round(100 * t / wall, 1)) if wall > 0 else (lambda t: 0.0)
    return {
        "epoch": epoch,
        "train_loss": round(loss_sum / steps, 4) if steps else 0.0,
        "val_loss": _round(val_loss, 4),
        "e2e_docs_s": _round(docs_per_s(docs, wall), 2),
        "compute_docs_s": _round(docs_per_s(docs, compute), 2),
        "val_docs_s": _round(val_docs_s, 2),
        "wall_s": round(wall, 3),
        "compute_s": round(compute, 3),
        "overhead_s": round(overhead, 3),
        "compute_pct": pct(compute),
        "overhead_pct": pct(overhead),
    }


def _print_epoch_table(records: list[dict]) -> None:
    """Pretty per-epoch summary: loss + every docs/s window (see README.md Metrics)."""
    table = PrettyTable()
    table.field_names = [
        "epoch",
        "train_loss",
        "val_loss",
        "e2e d/s",
        "compute d/s",
        "val d/s",
        "compute %",
        "overhead %",
    ]

    def f(v, fmt):
        return format(v, fmt) if v is not None else "-"

    for r in records:
        table.add_row(
            [
                r["epoch"],
                f(r["train_loss"], ".4f"),
                f(r["val_loss"], ".4f"),
                f(r["e2e_docs_s"], ".1f"),
                f(r["compute_docs_s"], ".1f"),
                f(r["val_docs_s"], ".1f"),
                f(r["compute_pct"], ".0f"),
                f(r["overhead_pct"], ".0f"),
            ]
        )
    print("\n── Per-epoch summary ──────────────────────────────")
    print(table)


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

    # Resolved run identity: names the metrics record AND the checkpoint subfolder
    # so weights and metrics for a run share one label.
    run_name = config.run_name or f"lr{config.lr}-bs{config.batch_size}"

    # --- mlflow ---
    if config.mlflow_experiment:
        mlflow.set_experiment(config.mlflow_experiment)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(asdict(config))

    # --- model + processor (single processor is the sole vocab source) ---
    if config.smoke:
        print("Loading tiny model (donut.synthetic, CPU, no downloads) ...")
    else:
        print(f"Loading model from {config.model_name} (backend={config.backend}) ...")
    model, processor = build_model(
        config.model_name,
        config.device,
        config.backend,
        smoke=config.smoke,
    )
    set_image_size(model, processor, config.image_size[0], config.image_size[1])

    # Confirm the donut optimizations are actually live in the training path, and
    # print the real attn impls (fact, not assumption) so the bench is interpretable.
    if not config.smoke:
        check_accel(model, config.backend)
    block = model.encoder.encoder.layers[0].blocks[0].attention.self
    print(
        f"  accel backend={config.backend}  "
        f"encoder_sdpa_patched={getattr(block, '_sdpa_patched', False)}  "
        f"decoder_attn={decoder_attn_impl(model)}"
    )

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

    train_ds = DonutDataset(train_samples, processor, config.max_length)
    val_ds = DonutDataset(val_samples, processor, config.max_length)
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

    log_mlflow = config.mlflow_experiment is not None
    epoch_records: list[dict] = []
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(config.max_epochs):
        # --- train one epoch ---
        # The epoch wall splits two ways: compute (synced fwd+bwd+clip+opt) and overhead
        # (everything else — H2D, scheduler.step, .item()/postfix/mlflow/sync). The whole
        # epoch is the e2e window. See README.md (Metrics) for why data% is not split out.
        model.train()
        compute = 0.0
        docs = steps = 0
        loss_sum = 0.0
        wall_start = time.perf_counter()

        bar = tqdm(
            train_loader, desc=f"epoch {epoch + 1}/{config.max_epochs}", leave=False
        )
        for batch in bar:
            # H2D stays outside the compute window (→ overhead) so compute_docs_s matches
            # bench_train_step, where tensors are already on the device.
            pixel_values = batch["pixel_values"].to(config.device)
            labels = batch["labels"].to(config.device)

            c0 = time.perf_counter()
            with autocast(config.device, config.precision):
                loss = model(pixel_values=pixel_values, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            _sync(config.device)
            compute += time.perf_counter() - c0

            scheduler.step()
            loss_sum += loss.item()
            docs += pixel_values.shape[0]
            steps += 1
            global_step += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")
            if log_mlflow:
                mlflow.log_metric("train_loss_step", loss.item(), step=global_step)

        wall = time.perf_counter() - wall_start

        # --- validation ---
        val_loss, val_docs_s = evaluate(
            model, val_loader, config.device, config.precision
        )

        rec = epoch_record(
            epoch + 1,
            wall=wall,
            compute=compute,
            docs=docs,
            steps=steps,
            loss_sum=loss_sum,
            val_loss=val_loss,
            val_docs_s=val_docs_s,
        )
        epoch_records.append(rec)

        val_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
        val_speed = f"{val_docs_s:.1f}" if val_docs_s is not None else "n/a"
        e2e = rec["e2e_docs_s"] or 0.0
        compute_ds = rec["compute_docs_s"] or 0.0
        print(
            f"Epoch {epoch + 1:3d}/{config.max_epochs}  "
            f"train={rec['train_loss']:.4f}  val={val_str}  | "
            f"e2e {e2e:.1f}  compute {compute_ds:.1f}  val {val_speed} doc/s  | "
            f"wall {rec['wall_s']:.1f}s = compute {rec['compute_s']:.1f}"
            f" + overhead {rec['overhead_s']:.1f}"
            f" ({rec['compute_pct']:.0f}%/{rec['overhead_pct']:.0f}%)"
        )
        if log_mlflow:
            mlflow.log_metrics(
                {k: v for k, v in rec.items() if k != "epoch" and v is not None},
                step=epoch + 1,
            )

        # --- checkpoint: keep the best-by-val-loss ---
        if not config.smoke and val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = Path(config.output_dir) / run_name / "best"
            save_checkpoint(model, processor, best_dir)
            print(f"           saved best (val={val_loss:.4f}) → {best_dir}")

    # --- checkpoint: final weights ---
    if not config.smoke:
        last_dir = Path(config.output_dir) / run_name / "last"
        save_checkpoint(model, processor, last_dir)
        print(f"Saved last → {last_dir}")

    # --- per-epoch summary table + developed JSON (named like the bench records) ---
    _print_epoch_table(epoch_records)
    if not config.smoke:
        # Weights live in config.output_dir (checkpoints/); the run record goes to
        # the shared results root alongside the bench/predict records.
        out_dir = GLOBAL_OUT_DIR / "results" / "train"
        name = f"train__{run_name}__{datetime.now():%Y%m%d-%H%M%S}.json"
        save_record(
            out_dir,
            name,
            {
                "meta": run_meta(config.device, None, config.model_name),
                "config": asdict(config),
                "best_val_loss": best_val_loss
                if best_val_loss != float("inf")
                else None,
                "epochs": epoch_records,
            },
        )
        print(f"Saved metrics → {out_dir / name}")

    print("\nTraining complete.")

    if log_mlflow:
        mlflow.end_run()


# ── CLI ─────────────────────────────────────────────────────────────────────--
app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_name: str = MODEL_ID,
    data_json: str = str(_REPO_ROOT / "test_data" / "train.json"),
    val_split: float = 0.2,
    # (height, width) fed to the encoder; lower = faster + less VRAM, less legible.
    image_height: int = DEFAULT_IMAGE_SIZE[0],
    image_width: int = DEFAULT_IMAGE_SIZE[1],
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 4,
    num_workers: int = 4,
    lr: float = 3e-4,
    warmup_steps: int = 100,
    max_epochs: int = 30,
    # Root of the per-run checkpoint dirs: <output_dir>/<run_name>/{best,last}.
    output_dir: Path = GLOBAL_OUT_DIR / "checkpoints",
    # Set an experiment name to enable MLflow logging; run_name defaults to lr+bs.
    mlflow_experiment: str | None = None,
    run_name: str | None = None,
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
