"""Shared CLI plumbing for the audit/bench scripts: arg parsing, device/dtype
resolution, run metadata, and result serialization."""

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


def parse_ints(s: str) -> list[int]:
    """Parse a comma-separated string of ints, e.g. "1,2,4" -> [1, 2, 4]."""
    return [int(tok.strip()) for tok in s.split(",") if tok.strip()]


def parse_image_sizes(s: str) -> list[tuple[int, int]]:
    """Parse "HxW,HxW" -> [(h, w), ...], e.g. "1280x960,1920x1440"."""
    sizes = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        h_str, w_str = token.split("x")
        sizes.append((int(h_str), int(w_str)))
    return sizes


def resolve_device_dtype(
    device: str | None, dtype: str | None
) -> tuple[str, torch.dtype]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype:
        torch_dtype = DTYPES[dtype]
    else:
        torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return device, torch_dtype


def run_meta(device: str | None, dtype: str | None, model_id: str) -> dict:
    device, torch_dtype = resolve_device_dtype(device, dtype)
    return {
        "model_id": model_id,
        "device": device,
        "dtype": str(torch_dtype).removeprefix("torch."),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_record(out_dir: Path, name: str, obj) -> None:
    """Write one self-describing record JSON into a results directory.

    Per-config files (rather than one monolithic file) let partial/repeated
    sweeps accumulate in the same directory — run small configs one day, large
    ones another, and the notebooks glob them all back together.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(obj, indent=2, default=str))
