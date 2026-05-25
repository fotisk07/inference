#!/usr/bin/env bash
# Exp03: Per-layer breakdown over 20 real images.
# Shows which encoder stages and decoder layers dominate latency.
# Encoder: absolute ms per stage. Decoder: ms per generated token per layer.
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

echo "=== Exp03: Layer Breakdown (20 images) ==="
uv run --project "$ROOT" bench_layers.py \
  --pool 20 \
  --n-images 20 \
  --save "$RESULTS/exp03_layer_breakdown.json"

echo "=== Exp03 done. Results in $RESULTS/exp03_layer_breakdown.json ==="
