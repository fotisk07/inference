import platform
import statistics
import sys

import torch
import transformers


def stat(vals: list[float]) -> dict:
    """Compute mean/std/p50/p95/p99. Used for per-run result summaries."""
    if len(vals) < 2:
        v = vals[0] if vals else 0.0
        return {
            "mean": round(v, 3),
            "std": 0.0,
            "p50": round(v, 3),
            "p95": round(v, 3),
            "p99": round(v, 3),
        }
    q = statistics.quantiles(vals, n=100, method="inclusive")
    return {
        "mean": round(statistics.mean(vals), 3),
        "std": round(statistics.stdev(vals), 3),
        "p50": round(q[49], 3),
        "p95": round(q[94], 3),
        "p99": round(q[98], 3),
    }


def fmt(values: list[float]) -> str:
    """Format a list of ms values as 'mean ± std ms'."""
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{m:7.1f} ± {s:5.1f} ms"


def system_info() -> dict:
    info: dict = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = str(torch.backends.cudnn.version())
    return info
