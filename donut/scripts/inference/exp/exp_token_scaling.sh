#!/usr/bin/env bash
# Experiment 2 -- decode-length scaling: vary max_new_tokens at fixed image
# size (1280x960) and batch size (1). This is where flash attention's regime
# argument lives -- every decode step is query_len=1, so growing the output
# length grows kv_len while query_len stays pinned. Shows the fa-vs-sdpa gap
# widening with more tokens. Capped at 1024 (< donut-base's 1536 position
# limit; larger crashes with an out-of-bounds position-embedding gather).
#
# Usage: bash scripts/exp/exp_token_scaling.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

BACKENDS="baseline,eager,sdpa,sdpa_flash,sdpa_math,sdpa_efficient,sdpa_cudnn,fa"

uv run python scripts/inference/bench_speed.py \
  --backends "$BACKENDS" \
  --image-sizes 1280x960 \
  --batch-sizes 1 \
  --max-new-tokens 8,32,128,512,1024 \
  --out results/exp_token_scaling \
  "$@"
