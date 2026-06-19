"""Shared CLI plumbing for the audit/bench scripts."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from donut.constants import MODEL_ID
from donut.model import load_model

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


def resolve_device_dtype(device, dtype) -> tuple[str, torch.dtype]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype:
        dtype = DTYPES[dtype]
    else:
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return device, dtype


def load_baseline_model(
    model_id, device: str, dtype: str, tiny: bool = False
) -> tuple[torch.nn.Module, str]:
    """Load the model with NO accelerations applied. Returns (model, model_id)."""
    device, dtype = resolve_device_dtype(device, dtype)
    if tiny:
        from donut.synthetic import make_tiny_model

        model = make_tiny_model(seed=0).to(device=device, dtype=dtype)
        model_id = "tiny-random-donut"
    else:
        model_id = model_id or MODEL_ID
        model, _ = load_model(model_id, device, dtype, backend="baseline")
        model.to(device)
    return model.eval(), model_id


def run_meta(device: str, dtype: str, model_id: str) -> dict:
    device, dtype = resolve_device_dtype(device, dtype)
    return {
        "model_id": model_id,
        "device": device,
        "dtype": str(dtype).removeprefix("torch."),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"wrote {path}")


def save_record(out_dir: Path, name: str, obj) -> None:
    """Write one self-describing record JSON into a results directory.

    Per-config files (rather than one monolithic file) let partial/repeated
    sweeps accumulate in the same directory — run small configs one day, large
    ones another, and the notebooks glob them all back together.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(obj, indent=2))
    print(f"wrote {path}")


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")
