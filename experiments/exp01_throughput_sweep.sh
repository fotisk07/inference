#!/usr/bin/env bash
# =============================================================================
# Experiment 1 — GPU Throughput Ceiling
# =============================================================================
# Goal:
#   Find the batch size that maximises docs/sec on a single GPU.
#   This is the primary experiment — every other experiment builds on it.
#
# Method:
#   Sweep batch sizes 1 → 64 with 100 measurement runs + 10 warmup per point.
#   OOM is handled gracefully: any batch size that does not fit in VRAM is
#   skipped and marked OOM in the sweep CSV.
#
# Key questions answered:
#   - What is the peak throughput (docs/sec) achievable on this GPU?
#   - At which batch size does throughput saturate?
#   - What is the memory cost of each batch size?
#   - How much does per-sample latency improve with batching?
#
# Outputs  →  logs/exp01_throughput_sweep/
#   *_sweep.csv          aggregated metrics per batch size (primary plotting input)
#   *_bs{N}_runs.csv     per-run detail for each batch size
#   *_sweep.json         full JSON with p50/p90/p95/p99 statistics
#
# Visualisations in analysis.ipynb  →  Section 1
#   - Throughput (docs/sec) + per-sample latency vs batch size (dual-axis)
#   - Peak GPU memory vs batch size
#   - Decoder efficiency vs batch size
#
# Estimated runtime:  30–60 min on a single A100/V100
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

uv run run_benchmark.py \
  --device cuda \
  --batch-sizes 1 2 4 8 16 32 64 \
  --num-runs 100 \
  --warmup-runs 10 \
  --pool-size 0 \
  --output-dir logs/exp01_throughput_sweep
