#!/usr/bin/env bash
# =============================================================================
# Experiment 6 — Sustained Throughput (Thermal & Long-Run Stability)
# =============================================================================
# Goal:
#   Detect throughput degradation during extended operation.  Real deployment
#   runs the model continuously — not just 100 warm benchmark iterations.
#   Thermal throttling (especially on consumer GPUs), CUDA memory
#   fragmentation, and driver scheduling jitter all show up only over time.
#
# Method:
#   Run 1000 consecutive measurement passes at two batch sizes (B=8 and B=16,
#   near the typical optimal range from exp01) with 20 warmup runs.
#   The per-run CSV has one row per iteration, so any temporal drift in
#   docs_per_second or latency is directly visible.
#
# Key questions answered:
#   - Does throughput remain constant or degrade over 1000 iterations?
#   - Is there a "break-in" period beyond the warmup window?
#   - Are there periodic latency spikes (GC, driver, thermal events)?
#   - What is the long-run stable throughput vs. the short-benchmark estimate?
#
# Outputs  →  logs/exp06_sustained/
#   *_bs8_runs.csv    1000-row latency time series at B=8
#   *_bs16_runs.csv   1000-row latency time series at B=16
#
# Visualisations in analysis.ipynb  →  Section 6
#   - Docs/sec time series over 1000 runs (B=8 and B=16)
#   - Rolling mean throughput (window=50) to smooth noise
#   - Latency distribution: first 100 vs last 100 runs (stability check)
#
# Estimated runtime:  3–5 h on a single A100 (schedule overnight)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python run_benchmark.py \
  --device cuda \
  --batch-sizes 8 16 \
  --num-runs 1000 \
  --warmup-runs 20 \
  --pool-size 0 \
  --output-dir logs/exp06_sustained
