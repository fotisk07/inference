#!/usr/bin/env bash
# Exp05: TTFT vs TPOT per-layer breakdown.
#
# Separates the first-token decode step (TTFT — no warm KV cache) from
# subsequent steps (TPOT — steady-state cached decode). Runs with more
# images than exp03 to get stable TTFT statistics, which have higher
# variance than TPOT due to the cold first step.
#
# Key questions:
#   - How much more expensive is the first token vs subsequent tokens?
#   - Which layers drive the TTFT overhead?
#   - What is the true steady-state TPOT once the KV cache is warm?
#
# Requires CUDA.
#
# Run directly on a GPU node:
#   bash experiments/exp05_ttft_tpot.sh
# Or via SLURM:
#   bash run_slurm.sh prod80 bash experiments/exp05_ttft_tpot.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

echo "=== Exp05: TTFT vs TPOT layer breakdown ==="
uv run --project "$ROOT" scripts/bench_layers.py \
  --pool 50 \
  --n-images 50 \
  --save "$RESULTS/exp05_ttft_tpot.json"

echo "=== Exp05 done. Results in $RESULTS/exp05_ttft_tpot.json ==="
