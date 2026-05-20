#!/usr/bin/env bash
# =============================================================================
# Experiment 4 — CPU Baseline
# =============================================================================
# Goal:
#   Establish a CPU-only baseline to quantify the GPU speedup factor and to
#   make the benchmark applicable to CPU-only deployment scenarios (edge
#   devices, containerised environments without GPU access).
#
# Method:
#   Sweep B=1,2,4 on CPU with 20 measurement runs + 2 warmup.
#   CPU inference is 10–50× slower so fewer runs still yield stable means.
#   Pool limited to 20 images to keep dataset-loading time negligible.
#   Note: FLOPs calibration is skipped on CPU (profiler overhead too high).
#
# Key questions answered:
#   - What is the GPU speedup over CPU at equivalent batch sizes?
#   - Does CPU benefit from batching in the same way GPU does?
#   - Is CPU inference viable for low-volume / cost-sensitive deployments?
#
# Outputs  →  logs/exp04_cpu_baseline/
#   *_sweep.csv          CPU aggregated metrics
#   *_bs{N}_runs.csv     CPU per-run latency detail
#
# Visualisations in analysis.ipynb  →  Section 4
#   - Side-by-side docs/sec bars: CPU vs GPU at B=1, 2, 4
#   - GPU speedup factor (GPU docs/sec ÷ CPU docs/sec)
#
# Estimated runtime:  60–180 min depending on CPU (consider running overnight)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python run_benchmark.py \
  --device cpu \
  --batch-sizes 1 2 4 \
  --num-runs 20 \
  --warmup-runs 2 \
  --pool-size 20 \
  --output-dir logs/exp04_cpu_baseline
