#!/usr/bin/env bash
# =============================================================================
# Experiment 3 — Phase-Level Bottleneck Analysis
# =============================================================================
# Goal:
#   Determine where inference time is spent (preprocessing, encoder, decoder,
#   postprocessing) at each batch size, and identify which phase limits
#   throughput scaling as batch size grows.
#
# Method:
#   Sweep B=1..32 with 300 measurement runs per point (3x more than exp01).
#   The higher run count ensures that per-phase p99 estimates are stable.
#   Batch sizes capped at 32 to stay reliably within VRAM limits.
#
# Key questions answered:
#   - Which phase (encoder vs decoder) dominates total latency?
#   - How does each phase scale with batch size — linearly or sub-linearly?
#   - At the optimal batch size, what fraction of time is "useful" compute?
#   - Where should optimisation effort be focused?
#
# Outputs  →  logs/exp03_phase_breakdown/
#   *_sweep.csv          aggregated per-phase stats per batch size
#   *_bs{N}_runs.csv     per-run preprocess/encoder/decoder/postprocess times
#
# Visualisations in analysis.ipynb  →  Section 3
#   - Stacked bar: mean phase times at each batch size
#   - Encoder vs decoder time growth curves (log-log)
#   - Phase fraction pie chart at the peak-throughput batch size
#
# Estimated runtime:  ~90 min on a single A100
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python run_benchmark.py \
  --device cuda \
  --batch-sizes 1 2 4 8 16 32 \
  --num-runs 300 \
  --warmup-runs 20 \
  --pool-size 0 \
  --output-dir logs/exp03_phase_breakdown
