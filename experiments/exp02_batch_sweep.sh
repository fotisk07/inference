#!/usr/bin/env bash
# Exp02: Throughput vs batch size sweep.
# Measures how samples/sec and tokens/sec scale as batch size grows.
# Decode time per batch is padded to the longest sequence in the batch, so
# throughput gains diminish. Memory usage also scales with batch size.
#
# Run directly on a GPU node:
#   bash experiments/exp02_batch_sweep.sh
# Or via SLURM:
#   bash run_slurm.sh prod80 bash experiments/exp02_batch_sweep.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

for BS in 1 2 4 8; do
  echo "=== Exp02: Batch size $BS ==="
  uv run --project "$ROOT" scripts/bench_dataset.py \
    --pool 50 \
    --runs 30 \
    --batch_size "$BS" \
    --save "$RESULTS/exp02_batch_bs${BS}.json"
done

echo "=== Exp02 done. Results in $RESULTS/exp02_batch_bs*.json ==="
