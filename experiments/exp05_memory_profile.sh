#!/usr/bin/env bash
# Exp05: Memory footprint vs batch size.
# Tracks peak GPU memory separately for the encode phase and decode phase.
# Encoder memory is dominated by activations (scales with batch × spatial size).
# Decoder memory grows as the KV-cache fills during generation.
#
# Run directly on a GPU node:
#   bash experiments/exp05_memory_profile.sh
# Or via SLURM:
#   bash run_slurm.sh prod80 bash experiments/exp05_memory_profile.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

for BS in 1 2 4 8; do
  echo "=== Exp05: Memory profile at batch_size=$BS ==="
  uv run --project "$ROOT" scripts/bench_dataset.py \
    --pool 20 \
    --runs 10 \
    --batch_size "$BS" \
    --save "$RESULTS/exp05_memory_bs${BS}.json"
    --no_patch
done

echo "=== Exp05 done. Results in $RESULTS/exp05_memory_bs*.json ==="
