#!/usr/bin/env bash
# Exp04: Validate that decode_ms scales linearly with sequence length.
# Runs the model with capped generation lengths (--max-new-tokens).
# If the ms/token metric is valid, decode_ms_per_token should be ~constant
# across all max_new_tokens values (since KV-cache makes each step O(1)).
# The intercept (latency at 1 token) captures first-token overhead.
#
# Run directly on a GPU node:
#   bash experiments/exp04_decoder_scaling.sh
# Or via SLURM:
#   bash run_slurm.sh prod80 bash experiments/exp04_decoder_scaling.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

for MAX_TOK in 10 25 50 100 200; do
  echo "=== Exp04: max-new-tokens=$MAX_TOK ==="
  uv run --project "$ROOT" bench_dataset.py \
    --pool 20 \
    --runs 20 \
    --batch-size 1 \
    --max-new-tokens "$MAX_TOK" \
    --save "$RESULTS/exp04_decoder_maxtok${MAX_TOK}.json"
done

echo "=== Exp04 done. Results in $RESULTS/exp04_decoder_maxtok*.json ==="
