#!/usr/bin/env bash
# =============================================================================
# Experiment 2 — Measurement Stability & Warmup Characterisation
# =============================================================================
# Goal:
#   Quantify measurement noise and determine the minimum number of warmup
#   iterations needed to reach a stable GPU performance state. This validates
#   the statistical assumptions (stationarity, normality) underlying all
#   other experiments.
#
# Method:
#   Run 1  (no_warmup)   — 500 measurement runs, 0 warmup at B=1.
#                          The GPU ramp-up curve is visible in the first runs.
#   Run 2  (with_warmup) — 500 measurement runs, 20 warmup at B=1.
#                          Establishes the stable-state reference distribution.
#   Both runs use the full 100-image pool to ensure result diversity.
#
# Key questions answered:
#   - How many warmup runs are needed before measurements stabilise?
#   - What is the coefficient of variation of end-to-end latency?
#   - Are 50 runs (the default) sufficient for stable p95/p99 estimates?
#   - Does the latency distribution follow a recognisable shape?
#
# Outputs  →  logs/exp02_stability/{no_warmup,with_warmup}/
#   *.csv    500-row per-run latency table for each variant
#   *.json   aggregated statistics + per-run records
#
# Visualisations in analysis.ipynb  →  Section 2
#   - Latency over run index (both variants overlaid)
#   - Rolling std convergence (window 10, 50, 100)
#   - Latency distribution: histogram + KDE comparison
#
# Estimated runtime:  ~25 min per run  (~50 min total) on a single A100
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Run 1/2: no warmup  (logs/exp02_stability/no_warmup/)"
uv run python run_benchmark.py \
  --device cuda \
  --batch-size 1 \
  --num-runs 500 \
  --warmup-runs 0 \
  --pool-size 0 \
  --output-dir logs/exp02_stability/no_warmup

echo "==> Run 2/2: 20 warmup  (logs/exp02_stability/with_warmup/)"
uv run python run_benchmark.py \
  --device cuda \
  --batch-size 1 \
  --num-runs 500 \
  --warmup-runs 20 \
  --pool-size 0 \
  --output-dir logs/exp02_stability/with_warmup
