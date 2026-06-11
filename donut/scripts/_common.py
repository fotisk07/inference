"""Shared CLI plumbing for the audit/bench scripts."""

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-id", default=None, help="HF model id (default: Donut CORD)")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto-detect)")
    p.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default=None,
        help="default: bf16 on cuda, f32 on cpu",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument(
        "--tiny",
        action="store_true",
        help="use the tiny random model (offline smoke run, no downloads)",
    )
    return p


def resolve_device_dtype(args) -> tuple[str, torch.dtype]:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype:
        dtype = DTYPES[args.dtype]
    else:
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return device, dtype


def load_baseline_model(args) -> tuple[torch.nn.Module, str]:
    """Load the model with NO accelerations applied. Returns (model, model_id)."""
    device, dtype = resolve_device_dtype(args)
    if args.tiny:
        from donut.synthetic import make_tiny_model

        model = make_tiny_model(seed=0).to(device=device, dtype=dtype)
        model_id = "tiny-random-donut"
    else:
        from transformers import VisionEncoderDecoderModel

        from donut.constants import MODEL_ID

        model_id = args.model_id or MODEL_ID
        model = VisionEncoderDecoderModel.from_pretrained(model_id, dtype=dtype)
        model = model.to(device)
    return model.eval(), model_id


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except Exception:
        return None


def run_meta(args, model_id: str) -> dict:
    device, dtype = resolve_device_dtype(args)
    return {
        "model_id": model_id,
        "device": device,
        "dtype": str(dtype).removeprefix("torch."),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "seed": args.seed,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
    }


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"wrote {path}")


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")
