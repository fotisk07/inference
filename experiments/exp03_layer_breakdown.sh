#!/usr/bin/env bash
# Exp03: Per-layer timing breakdown at batch=1.
#
# Times each Swin encoder stage and each MBart decoder layer using CUDA events.
# Produces relative % contribution per layer and mean ± std across images.
# Compare with exp01 for the high-level component picture.
#
# Requires CUDA.
#
# Run directly on a GPU node:
#   bash experiments/exp03_layer_breakdown.sh
# Or via SLURM:
#   bash run_slurm.sh prod80 bash experiments/exp03_layer_breakdown.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

echo "=== Exp03: Layer-level breakdown (batch=1) ==="
uv run --project "$ROOT" scripts/bench_layers.py \
  --pool 50 \
  --n-images 20 \
  --save "$RESULTS/exp03_layer_breakdown.json"

echo "=== Exp03 done. Results in $RESULTS/exp03_layer_breakdown.json ==="
