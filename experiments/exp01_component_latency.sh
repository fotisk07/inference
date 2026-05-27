#!/usr/bin/env bash
# Exp01: High-level component latency breakdown at batch=1.
# Gives the most representative per-component timing: preprocess, encode,
# decode (ms/token), over 50 real images from the CORD-v2 test set.
#
# Run directly on a GPU node:
#   bash experiments/exp01_component_latency.sh
# Or via SLURM:
#   bash run_slurm.sh prod80 bash experiments/exp01_component_latency.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

echo "=== Exp01: Component Latency Breakdown (batch=1) ==="
uv run --project "$ROOT" scripts/bench_dataset.py \
  --pool 50 \
  --runs 50 \
  --batch_size 1 \
  --save "$RESULTS/exp01_component_latency.json"
