"""Generate predictions from a fine-tuned Donut checkpoint.

Orchestration only: model loading is donut.model.load_model (a checkpoint dir is just
a model_id), output parsing is donut.dataset, scoring is donut.metrics. This CLI wires
them together and owns the progress bar.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import typer
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from donut.constants import DEFAULT_MAX_NEW_TOKENS, GLOBAL_OUT_DIR
from donut.dataset import load_samples, parse_prediction
from donut.metrics import summarize
from donut.model import decoder_start_ids, load_model
from donut.runio import run_meta, save_record

# Repo root (…/inference) so the default --data-json resolves no matter the CWD.
_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Config:
    """Typed bundle of inference settings, built from the CLI by `main` below."""

    checkpoint: str  # checkpoint dir saved by train.py (e.g. checkpoints/best)
    data_json: str
    out: Path  # directory for the metrics record JSON (named like bench records)
    output_json: str | None  # optional: per-document {image, gt, pred} debug dump
    backend: str
    max_new_tokens: int
    batch_size: int
    num_workers: int
    device: str


# ── Inference loop ────────────────────────────────────────────────────────────


class _PredictDataset(Dataset):
    """Returns (pixel_values, sample) so the batched loop keeps the raw sample —
    image path + gt fields — alongside the tensor. (DonutDataset drops both, only
    emitting pixel_values + tokenized labels, which scoring/JSON can't use.)
    """

    def __init__(self, samples: list[dict], processor):
        self.samples = samples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(
            0
        )
        return pixel_values, sample


def _collate(batch: list[tuple]) -> tuple[torch.Tensor, list[dict]]:
    """Stack the (fixed-size) pixel tensors; keep samples as a plain list."""
    pixel_values, samples = zip(*batch)
    return torch.stack(pixel_values), list(samples)


def _gt_fields(sample: dict) -> dict[str, str]:
    """Ground truth — may be absent (inference-only mode)."""
    return {
        f["field_name"].split("/")[-1]: f.get("annotator_text", "").strip()
        for f in sample.get("fields", [])
        if f.get("annotator_text", "").strip()
    }


def run_predictions(
    model,
    processor,
    samples: list[dict],
    *,
    device: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = 8,
    num_workers: int = 4,
    progress: bool = True,
) -> list[dict]:
    """Generate field predictions for every sample.

    Returns [{"image", "gt": {field: value}, "pred": {field: value}}] — used both
    for donut.metrics scoring and (verbatim) as the saved --output_json. shuffle=False
    keeps each row's gt/pred paired with its source sample.
    """
    model.eval()
    model_dtype = next(model.parameters()).dtype

    loader = DataLoader(
        _PredictDataset(samples, processor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )

    results = []
    batches = tqdm(loader, desc="predicting") if progress else loader
    for pixel_values, batch_samples in batches:
        pixel_values = pixel_values.to(device=device, dtype=model_dtype)
        # Canonical Donut: the task token is the decoder start (training set
        # decoder_start_token_id = <s_donut>), so seeding generation with it matches
        # the training-time decoder input position-for-position.
        decoder_input_ids = decoder_start_ids(model, batch_size=pixel_values.size(0))

        with torch.no_grad():
            output_ids = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=max_new_tokens,
            )

        # token2json ignores the pad/eos that batched generate right-pads with, so
        # decoding the whole row (special tokens kept, to keep the <s_field> markers)
        # is safe per-sample.
        for ids, sample in zip(output_ids, batch_samples):
            decoded = processor.tokenizer.decode(ids, skip_special_tokens=False)
            results.append(
                {
                    "image": str(sample["image"]),
                    "gt": _gt_fields(sample),
                    "pred": parse_prediction(decoded, processor),
                }
            )

    return results


def predict(cfg: Config) -> None:
    print(f"Checkpoint : {cfg.checkpoint}")
    print(f"Data       : {cfg.data_json}")
    print(f"Backend    : {cfg.backend}  device={cfg.device}\n")

    model, processor = load_model(
        model_id=cfg.checkpoint, device=cfg.device, backend=cfg.backend
    )

    samples = load_samples(Path(cfg.data_json))
    print(f"Processing {len(samples)} samples ...")

    results = run_predictions(
        model,
        processor,
        samples,
        device=cfg.device,
        max_new_tokens=cfg.max_new_tokens,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # Metrics — only when GT is present. summarize() prints a PrettyTable and
    # returns the data dict; persist both modes for later notebook analysis.
    if any(r["gt"] for r in results):
        meta = run_meta(cfg.device, None, cfg.checkpoint)
        meta["dtype"] = str(next(model.parameters()).dtype).removeprefix("torch.")
        meta["backend"] = cfg.backend
        record = {
            "meta": meta,
            "config": asdict(cfg),
            "n_samples": len(samples),
            "metrics": {
                "strict": summarize(results, soft=False),
                "soft": summarize(results, soft=True),
            },
        }
        name = f"predict__{Path(cfg.checkpoint).name}__{Path(cfg.data_json).stem}__{datetime.now():%Y%m%d-%H%M%S}.json"
        save_record(cfg.out, name, record)
        print(f"Saved metrics → {cfg.out / name}")

    # Optional debug dump — per-document {image, gt, pred} records
    if cfg.output_json:
        out_path = Path(cfg.output_json)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Saved predictions → {out_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────--
app = typer.Typer(add_completion=False)


@app.command()
def main(
    # Checkpoint dir saved by train.py (e.g. checkpoints/best or checkpoints/last).
    checkpoint: str,
    data_json: str = str(_REPO_ROOT / "test_data" / "train.json"),
    out: Path = GLOBAL_OUT_DIR / "results" / "predict",
    output_json: str | None = typer.Option(
        None,
        help="optional: also write per-document {image, gt, pred} predictions here (debug)",
    ),
    backend: str = "sdpa",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = 8,
    num_workers: int = 4,
    device: str | None = None,
) -> None:
    """Score a fine-tuned Donut checkpoint on labelled data."""
    predict(
        Config(
            checkpoint=checkpoint,
            data_json=data_json,
            out=out,
            output_json=output_json,
            backend=backend,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
    )


if __name__ == "__main__":
    app()
