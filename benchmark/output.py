from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .metrics import AggregatedResult, BatchSweepResult, MetricStats, RunResult


def setup_logging(output_dir: str, run_id: str) -> logging.Logger:
    """Configure root 'benchmark' logger with console (INFO) and file (DEBUG) handlers."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log_path = os.path.join(output_dir, f"{run_id}.log")
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    root = logging.getLogger("benchmark")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(file_handler)

    return root


def _stats_to_dict(s: MetricStats) -> dict[str, float]:
    return asdict(s)


def _run_to_dict(r: RunResult) -> dict[str, Any]:
    return {
        "run_index": r.run_index,
        "batch_size": r.batch_size,
        "image_indices": r.image_indices,
        "preprocess_ms": r.preprocess_ms,
        "encoder_ms": r.encoder_ms,
        "decoder_ms": r.decoder_ms,
        "postprocess_ms": r.postprocess_ms,
        "end_to_end_ms": r.end_to_end_ms,
        "per_sample_latency_ms": r.per_sample_latency_ms,
        "docs_per_second": r.docs_per_second,
        "num_generated_tokens": r.num_generated_tokens,
        "max_generated_tokens_in_batch": r.max_generated_tokens_in_batch,
        "tokens_per_second": r.tokens_per_second,
        "decoder_efficiency": r.decoder_efficiency,
        "orig_image_width_mean": r.orig_image_width_mean,
        "orig_image_height_mean": r.orig_image_height_mean,
        "orig_megapixels_mean": r.orig_megapixels_mean,
        "processed_image_width": r.processed_image_width,
        "processed_image_height": r.processed_image_height,
        "processed_megapixels": r.processed_megapixels,
        "orig_megapixels_per_second": r.orig_megapixels_per_second,
        "processed_megapixels_per_second": r.processed_megapixels_per_second,
        "peak_gpu_memory_mb": r.peak_gpu_memory_mb,
        "allocated_gpu_memory_mb": r.allocated_gpu_memory_mb,
        "reserved_gpu_memory_mb": r.reserved_gpu_memory_mb,
        "total_flops": r.total_flops,
        "encoder_tflops": r.encoder_tflops,
        "decoder_tflops": r.decoder_tflops,
        "is_valid_json": r.is_valid_json,
        "raw_output": r.raw_output,
    }


def _agg_section(result: AggregatedResult) -> dict[str, Any]:
    return {
        "preprocess_ms": _stats_to_dict(result.preprocess_ms),
        "encoder_ms": _stats_to_dict(result.encoder_ms),
        "decoder_ms": _stats_to_dict(result.decoder_ms),
        "postprocess_ms": _stats_to_dict(result.postprocess_ms),
        "end_to_end_ms": _stats_to_dict(result.end_to_end_ms),
        "per_sample_latency_ms": _stats_to_dict(result.per_sample_latency_ms),
        "docs_per_second": _stats_to_dict(result.docs_per_second),
        "tokens_per_second": _stats_to_dict(result.tokens_per_second),
        "decoder_efficiency": _stats_to_dict(result.decoder_efficiency),
        "peak_gpu_memory_mb": _stats_to_dict(result.peak_gpu_memory_mb),
        "encoder_tflops": _stats_to_dict(result.encoder_tflops),
        "decoder_tflops": _stats_to_dict(result.decoder_tflops),
        "orig_image_width_mean": _stats_to_dict(result.orig_image_width_mean),
        "orig_image_height_mean": _stats_to_dict(result.orig_image_height_mean),
        "orig_megapixels_mean": _stats_to_dict(result.orig_megapixels_mean),
        "processed_megapixels": _stats_to_dict(result.processed_megapixels),
        "orig_megapixels_per_second": _stats_to_dict(result.orig_megapixels_per_second),
        "processed_megapixels_per_second": _stats_to_dict(
            result.processed_megapixels_per_second
        ),
        "processed_image_width": result.processed_image_width,
        "processed_image_height": result.processed_image_height,
    }


def write_json(result: AggregatedResult, path: str) -> None:
    doc = {
        "run_id": result.run_id,
        "timestamp_utc": result.timestamp_utc,
        "config": asdict(result.config),
        "system_info": result.system_info,
        "baseline_gpu_allocated_mb": result.baseline_gpu_allocated_mb,
        "aggregated": _agg_section(result),
        "summary": {
            "valid_json_count": result.valid_json_count,
            "valid_json_rate": result.valid_json_rate,
            "oom_count": result.oom_count,
            "successful_runs": len(result.runs),
        },
        "runs": [_run_to_dict(r) for r in result.runs],
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    logging.getLogger("benchmark.output").info("Wrote %s", path)


_CSV_COLUMNS = [
    "run_index",
    "batch_size",
    "image_indices",
    "preprocess_ms",
    "encoder_ms",
    "decoder_ms",
    "postprocess_ms",
    "end_to_end_ms",
    "per_sample_latency_ms",
    "docs_per_second",
    "num_generated_tokens",
    "max_generated_tokens_in_batch",
    "tokens_per_second",
    "decoder_efficiency",
    "orig_image_width_mean",
    "orig_image_height_mean",
    "orig_megapixels_mean",
    "processed_image_width",
    "processed_image_height",
    "processed_megapixels",
    "orig_megapixels_per_second",
    "processed_megapixels_per_second",
    "peak_gpu_memory_mb",
    "allocated_gpu_memory_mb",
    "reserved_gpu_memory_mb",
    "total_flops",
    "encoder_tflops",
    "decoder_tflops",
    "is_valid_json",
]


def write_csv(runs: list[RunResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for r in runs:
            row = _run_to_dict(r)
            # image_indices is a list; stringify for CSV
            row["image_indices"] = str(row["image_indices"])
            writer.writerow({k: row[k] for k in _CSV_COLUMNS})
    logging.getLogger("benchmark.output").info("Wrote %s", path)


def write_sweep_json(sweep: BatchSweepResult, path: str) -> None:
    batch_sizes_data: dict[str, Any] = {}
    for B, agg in sweep.results_by_batch_size.items():
        if agg is None:
            batch_sizes_data[str(B)] = "OOM"
        else:
            batch_sizes_data[str(B)] = {
                "aggregated": _agg_section(agg),
                "summary": {
                    "valid_json_count": agg.valid_json_count,
                    "valid_json_rate": agg.valid_json_rate,
                    "oom_count": agg.oom_count,
                    "successful_runs": len(agg.runs),
                },
                "runs": [_run_to_dict(r) for r in agg.runs],
            }
    doc = {
        "run_id": sweep.run_id,
        "timestamp_utc": sweep.timestamp_utc,
        "config": asdict(sweep.config),
        "system_info": sweep.system_info,
        "batch_sizes": batch_sizes_data,
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    logging.getLogger("benchmark.output").info("Wrote sweep JSON: %s", path)


_SWEEP_CSV_COLUMNS = [
    "batch_size",
    "per_sample_latency_ms_mean",
    "per_sample_latency_ms_p99",
    "docs_per_second_mean",
    "tokens_per_second_mean",
    "decoder_efficiency_mean",
    "orig_megapixels_mean_mean",
    "orig_megapixels_per_second_mean",
    "processed_megapixels_per_second_mean",
    "processed_image_width",
    "processed_image_height",
    "peak_gpu_memory_mb_mean",
    "encoder_tflops_mean",
    "decoder_tflops_mean",
    "oom_count",
    "successful_runs",
]


def write_sweep_csv(sweep: BatchSweepResult, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SWEEP_CSV_COLUMNS)
        writer.writeheader()
        for B in sorted(sweep.results_by_batch_size):
            agg = sweep.results_by_batch_size[B]
            if agg is None:
                writer.writerow(
                    {
                        "batch_size": B,
                        **{k: "OOM" for k in _SWEEP_CSV_COLUMNS if k != "batch_size"},
                    }
                )
            else:
                writer.writerow(
                    {
                        "batch_size": B,
                        "per_sample_latency_ms_mean": agg.per_sample_latency_ms.mean,
                        "per_sample_latency_ms_p99": agg.per_sample_latency_ms.p99,
                        "docs_per_second_mean": agg.docs_per_second.mean,
                        "tokens_per_second_mean": agg.tokens_per_second.mean,
                        "decoder_efficiency_mean": agg.decoder_efficiency.mean,
                        "orig_megapixels_mean_mean": agg.orig_megapixels_mean.mean,
                        "orig_megapixels_per_second_mean": agg.orig_megapixels_per_second.mean,
                        "processed_megapixels_per_second_mean": agg.processed_megapixels_per_second.mean,
                        "processed_image_width": agg.processed_image_width,
                        "processed_image_height": agg.processed_image_height,
                        "peak_gpu_memory_mb_mean": agg.peak_gpu_memory_mb.mean,
                        "encoder_tflops_mean": agg.encoder_tflops.mean,
                        "decoder_tflops_mean": agg.decoder_tflops.mean,
                        "oom_count": agg.oom_count,
                        "successful_runs": len(agg.runs),
                    }
                )
    logging.getLogger("benchmark.output").info("Wrote sweep CSV: %s", path)


def print_summary(result: AggregatedResult) -> None:
    """Print a human-readable per-metric summary table."""
    sep = "-" * 70

    def _row(label: str, stats: MetricStats) -> str:
        return (
            f"  {label:<26} mean={stats.mean:>9.3f}  "
            f"p50={stats.p50:>9.3f}  p95={stats.p95:>9.3f}  std={stats.std:>8.3f}"
        )

    print(f"\n{'=' * 70}")
    print(f"  Donut Benchmark — {result.run_id}")
    print(sep)
    print(f"  Model : {result.config.model_id}")
    print(f"  Device: {result.config.device}   Batch size: {result.config.batch_size}")
    pool_desc = result.config.pool_size if result.config.pool_size else "all"
    print(
        f"  Pool  : {result.config.dataset_name}[{result.config.dataset_split}] pool_size={pool_desc}"
    )
    print(
        f"  Proc  : {result.processed_image_width:.0f}x{result.processed_image_height:.0f} px"
        f"  ({result.processed_megapixels.mean:.2f} MP processed by model)"
    )
    print(
        f"  Orig  : {result.orig_image_width_mean.mean:.0f}x{result.orig_image_height_mean.mean:.0f} px"
        f"  mean ({result.orig_megapixels_mean.mean:.2f} MP mean input)"
    )
    print(
        f"  Runs  : {len(result.runs)}/{result.config.num_runs} successful   OOM: {result.oom_count}"
    )
    print(
        f"  Valid JSON: {result.valid_json_count}/{len(result.runs)} ({result.valid_json_rate:.1%})"
    )
    print(sep)
    print(f"  {'Metric':<26} {'mean':>13}  {'p50':>13}  {'p95':>13}  {'std':>12}")
    print(sep)
    print(_row("preprocess       (ms)", result.preprocess_ms))
    print(_row("encoder          (ms)", result.encoder_ms))
    print(_row("decoder          (ms)", result.decoder_ms))
    print(_row("postprocess      (ms)", result.postprocess_ms))
    print(_row("end-to-end       (ms)", result.end_to_end_ms))
    print(_row("per-sample lat   (ms)", result.per_sample_latency_ms))
    print(_row("docs/sec              ", result.docs_per_second))
    print(_row("tokens/sec            ", result.tokens_per_second))
    print(_row("decoder efficiency    ", result.decoder_efficiency))
    print(_row("peak GPU mem     (MB) ", result.peak_gpu_memory_mb))
    print(_row("encoder TFLOPS        ", result.encoder_tflops))
    print(_row("decoder TFLOPS        ", result.decoder_tflops))
    print(_row("orig MP/sec           ", result.orig_megapixels_per_second))
    print(_row("proc MP/sec           ", result.processed_megapixels_per_second))
    print(f"{'=' * 70}\n")


def print_sweep_summary(sweep: BatchSweepResult) -> None:
    """Print a compact comparison table across all batch sizes."""
    print(f"\n{'=' * 90}")
    print(f"  Donut Batch-Size Sweep — {sweep.run_id}")
    print(f"  Model: {sweep.config.model_id}   Device: {sweep.config.device}")
    print("-" * 90)
    hdr = (
        f"  {'Batch':>5} | {'Lat/doc(ms)':>11} | {'Docs/sec':>8} | {'Tok/sec':>8} |"
        f" {'DecEff':>7} | {'PeakMem(MB)':>11} | {'EncTFLOPS':>10} | {'DecTFLOPS':>10}"
    )
    print(hdr)
    print("-" * 90)
    for B in sorted(sweep.results_by_batch_size):
        agg = sweep.results_by_batch_size[B]
        if agg is None:
            print(
                f"  {B:>5} | {'OOM':>11} | {'OOM':>8} | {'OOM':>8} |"
                f" {'OOM':>7} | {'OOM':>11} | {'OOM':>10} | {'OOM':>10}"
            )
        else:
            a = agg
            print(
                f"  {B:>5} | {a.per_sample_latency_ms.mean:>11.1f} |"
                f" {a.docs_per_second.mean:>8.2f} | {a.tokens_per_second.mean:>8.1f} |"
                f" {a.decoder_efficiency.mean:>7.3f} | {a.peak_gpu_memory_mb.mean:>11.1f} |"
                f" {a.encoder_tflops.mean:>10.4f} | {a.decoder_tflops.mean:>10.4f}"
            )
    print(f"{'=' * 90}\n")
