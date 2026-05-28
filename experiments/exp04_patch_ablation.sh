#!/usr/bin/env bash
# Exp04: Patch ablation — patched vs no-patch inference, back-to-back.
#
# Runs bench_dataset and bench_layers under both conditions in the same
# session so GPU state, image pool, and run count are identical.
# Quantifies the runtime benefit of the attention mask optimization
# across warm inference.
#
# Requires CUDA.
#
# Run directly on a GPU node:
#   bash experiments/exp04_patch_ablation.sh
# Or via SLURM:
#   bash run_slurm.sh prod80 bash experiments/exp04_patch_ablation.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

echo "=== Exp04a: Component latency — patched ==="
uv run --project "$ROOT" scripts/bench_dataset.py \
  --pool 50 \
  --runs 50 \
  --batch_size 1 \
  --save "$RESULTS/exp04_patched_dataset.json"

echo "=== Exp04b: Component latency — no-patch ==="
uv run --project "$ROOT" scripts/bench_dataset.py \
  --pool 50 \
  --runs 50 \
  --batch_size 1 \
  --no-patch \
  --save "$RESULTS/exp04_nopatch_dataset.json"

echo "=== Exp04c: Layer breakdown — patched ==="
uv run --project "$ROOT" scripts/bench_layers.py \
  --pool 50 \
  --n-images 20 \
  --save "$RESULTS/exp04_patched_layers.json"

echo "=== Exp04d: Layer breakdown — no-patch ==="
uv run --project "$ROOT" scripts/bench_layers.py \
  --pool 50 \
  --n-images 20 \
  --no-patch \
  --save "$RESULTS/exp04_nopatch_layers.json"

echo "=== Exp04 done. Results in $RESULTS/exp04_*.json ==="
