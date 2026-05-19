from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .config import BenchmarkConfig


@dataclass
class RunResult:
    run_index: int
    # --- workload identity ---
    batch_size: int
    image_indices: list[int]  # which pool images were in this batch

    # --- phase latencies (ms) ---
    preprocess_ms: float
    encoder_ms: float
    decoder_ms: float
    postprocess_ms: float
    end_to_end_ms: float

    # --- throughput ---
    per_sample_latency_ms: float  # end_to_end_ms / batch_size
    docs_per_second: float  # batch_size * 1000 / end_to_end_ms

    # --- token statistics ---
    num_generated_tokens: int  # sum of actual tokens across all samples
    max_generated_tokens_in_batch: (
        int  # longest sequence = number of decoder steps taken
    )
    tokens_per_second: float  # num_generated_tokens / (decoder_ms / 1000)
    # Actual / (B * max): how much decoder compute was useful vs padding waste.
    # Always 1.0 at B=1; < 1.0 for diverse batches with variable output lengths.
    decoder_efficiency: float

    # --- image dimensions ---
    orig_image_width_mean: float  # mean original PIL image width (px)
    orig_image_height_mean: float  # mean original PIL image height (px)
    orig_megapixels_mean: float  # mean per-image MP before processing (W*H/1e6)
    processed_image_width: int  # tensor width after DonutImageProcessor
    processed_image_height: int  # tensor height after DonutImageProcessor
    processed_megapixels: float  # processed_image_width * processed_image_height / 1e6
    orig_megapixels_per_second: float  # orig_megapixels_mean * B * 1000 / end_to_end_ms
    processed_megapixels_per_second: (
        float  # processed_megapixels * B * 1000 / end_to_end_ms
    )

    # --- GPU memory ---
    peak_gpu_memory_mb: float
    allocated_gpu_memory_mb: float
    reserved_gpu_memory_mb: float

    # --- FLOPs (lower-bound estimates from torch.profiler calibration) ---
    total_flops: int
    encoder_tflops: float
    decoder_tflops: float

    # --- output quality ---
    is_valid_json: bool  # True if every sample in batch parsed
    raw_output: str  # concatenation of per-sample outputs (debug)


@dataclass
class MetricStats:
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p90: float
    p95: float
    p99: float


def _default_stats() -> MetricStats:
    return MetricStats(0, 0, 0, 0, 0, 0, 0, 0)


@dataclass
class AggregatedResult:
    config: BenchmarkConfig
    runs: list[RunResult]

    # 8-stat blocks for every float metric in RunResult
    preprocess_ms: MetricStats = field(default_factory=_default_stats)
    encoder_ms: MetricStats = field(default_factory=_default_stats)
    decoder_ms: MetricStats = field(default_factory=_default_stats)
    postprocess_ms: MetricStats = field(default_factory=_default_stats)
    end_to_end_ms: MetricStats = field(default_factory=_default_stats)
    per_sample_latency_ms: MetricStats = field(default_factory=_default_stats)
    docs_per_second: MetricStats = field(default_factory=_default_stats)
    tokens_per_second: MetricStats = field(default_factory=_default_stats)
    decoder_efficiency: MetricStats = field(default_factory=_default_stats)
    peak_gpu_memory_mb: MetricStats = field(default_factory=_default_stats)
    encoder_tflops: MetricStats = field(default_factory=_default_stats)
    decoder_tflops: MetricStats = field(default_factory=_default_stats)
    orig_image_width_mean: MetricStats = field(default_factory=_default_stats)
    orig_image_height_mean: MetricStats = field(default_factory=_default_stats)
    orig_megapixels_mean: MetricStats = field(default_factory=_default_stats)
    processed_megapixels: MetricStats = field(default_factory=_default_stats)
    orig_megapixels_per_second: MetricStats = field(default_factory=_default_stats)
    processed_megapixels_per_second: MetricStats = field(default_factory=_default_stats)
    processed_image_width: float = 0.0
    processed_image_height: float = 0.0

    valid_json_count: int = 0
    valid_json_rate: float = 0.0
    oom_count: int = 0
    baseline_gpu_allocated_mb: float = 0.0
    timestamp_utc: str = ""
    system_info: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""


@dataclass
class BatchSweepResult:
    run_id: str
    config: BenchmarkConfig
    system_info: dict[str, Any]
    timestamp_utc: str
    # Keyed by batch_size. None means that batch size hit OOM on every run.
    results_by_batch_size: dict[int, AggregatedResult | None]


def _stats(values: list[float]) -> MetricStats:
    arr = np.array(values, dtype=np.float64)
    return MetricStats(
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        p50=float(np.percentile(arr, 50)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
    )


def _system_info() -> dict[str, Any]:
    try:
        import torch

        cuda_ver = torch.version.cuda or "n/a"
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"
        torch_ver = torch.__version__
    except Exception:
        cuda_ver = gpu_name = torch_ver = "n/a"

    try:
        import transformers

        transformers_ver = transformers.__version__
    except Exception:
        transformers_ver = "n/a"

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch_ver,
        "cuda_version": cuda_ver,
        "gpu_name": gpu_name,
        "transformers_version": transformers_ver,
        "numpy_version": np.__version__,
    }


def compute_stats(
    config: BenchmarkConfig,
    runs: list[RunResult],
    oom_count: int,
    baseline_gpu_allocated_mb: float,
    run_id: str,
) -> AggregatedResult:
    if not runs:
        raise ValueError("No successful runs to aggregate.")

    now = datetime.now(timezone.utc)

    return AggregatedResult(
        config=config,
        runs=runs,
        preprocess_ms=_stats([r.preprocess_ms for r in runs]),
        encoder_ms=_stats([r.encoder_ms for r in runs]),
        decoder_ms=_stats([r.decoder_ms for r in runs]),
        postprocess_ms=_stats([r.postprocess_ms for r in runs]),
        end_to_end_ms=_stats([r.end_to_end_ms for r in runs]),
        per_sample_latency_ms=_stats([r.per_sample_latency_ms for r in runs]),
        docs_per_second=_stats([r.docs_per_second for r in runs]),
        tokens_per_second=_stats([r.tokens_per_second for r in runs]),
        decoder_efficiency=_stats([r.decoder_efficiency for r in runs]),
        peak_gpu_memory_mb=_stats([r.peak_gpu_memory_mb for r in runs]),
        encoder_tflops=_stats([r.encoder_tflops for r in runs]),
        decoder_tflops=_stats([r.decoder_tflops for r in runs]),
        orig_image_width_mean=_stats([r.orig_image_width_mean for r in runs]),
        orig_image_height_mean=_stats([r.orig_image_height_mean for r in runs]),
        orig_megapixels_mean=_stats([r.orig_megapixels_mean for r in runs]),
        processed_megapixels=_stats([r.processed_megapixels for r in runs]),
        orig_megapixels_per_second=_stats([r.orig_megapixels_per_second for r in runs]),
        processed_megapixels_per_second=_stats(
            [r.processed_megapixels_per_second for r in runs]
        ),
        processed_image_width=float(runs[0].processed_image_width),
        processed_image_height=float(runs[0].processed_image_height),
        valid_json_count=sum(1 for r in runs if r.is_valid_json),
        valid_json_rate=sum(1 for r in runs if r.is_valid_json) / len(runs),
        oom_count=oom_count,
        baseline_gpu_allocated_mb=baseline_gpu_allocated_mb,
        timestamp_utc=now.isoformat(),
        system_info=_system_info(),
        run_id=run_id,
    )
