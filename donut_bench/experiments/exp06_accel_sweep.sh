#!/usr/bin/env bash
# Exp06: Acceleration backend comparison sweep.
# Runs verify_accel.py first to confirm correctness, then benchmarks
# each backend (eager, sdpa, fa2) at batch_size 1 and 4.
#
# Compare results against exp01 (baseline) to measure speedup.
#
# Run directly on a GPU node:
#   bash experiments/exp06_accel_sweep.sh
# Or via SLURM:
#   bash run_slurm.sh prod80 bash experiments/exp06_accel_sweep.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

echo "=== Exp06: Acceleration Backend Sweep ==="

echo ""
echo "--- Correctness checks ---"
uv run --project "$ROOT" scripts/verify_accel.py --backend sdpa --n_images 10
uv run --project "$ROOT" scripts/verify_accel.py --backend fa2 --n_images 10

echo ""
echo "--- Benchmarks ---"
for BACKEND in eager sdpa fa2; do
  for BS in 1 4; do
    echo ""
    echo "Backend=$BACKEND  batch_size=$BS"
    uv run --project "$ROOT" scripts/bench_accel.py \
      --backend "$BACKEND" \
      --pool 50 \
      --runs 50 \
      --batch_size "$BS" \
      --save "$RESULTS/exp06_${BACKEND}_bs${BS}.json"
  done
done

echo ""
echo "=== Exp06 complete. Results in $RESULTS/exp06_*.json ==="
