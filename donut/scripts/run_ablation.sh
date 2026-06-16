#!/usr/bin/env bash
# Donut inference speed ablation: backend × batch_size × image_resolution.
# Results land in <OUT>/bench_speed.json for analysis in notebooks/ablation.ipynb.
#
# Override any variable via the environment:
#   IMAGE_SIZES=640x480,1280x960 N_RUNS=50 bash run_ablation.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_SIZES="${IMAGE_SIZES:-640x480,960x720,1280x960}"
BACKENDS="${BACKENDS:-eager,sdpa,fa}"
BATCH_SIZES="${BATCH_SIZES:-1,2,4,8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
N_RUNS="${N_RUNS:-30}"
N_WARMUP="${N_WARMUP:-5}"
OUT="${OUT:-results/ablation}"

echo "=== Donut inference ablation ==="
echo "  image_sizes    = $IMAGE_SIZES"
echo "  backends       = $BACKENDS"
echo "  batch_sizes    = $BATCH_SIZES"
echo "  max_new_tokens = $MAX_NEW_TOKENS"
echo "  n_runs / warmup = $N_RUNS / $N_WARMUP"
echo "  output         = $OUT"
echo ""

uv run python bench_speed.py \
  --image-sizes  "$IMAGE_SIZES" \
  --backends     "$BACKENDS" \
  --batch-sizes  "$BATCH_SIZES" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --n-runs       "$N_RUNS" \
  --n-warmup     "$N_WARMUP" \
  --out          "$OUT"

echo ""
echo "=== Done — open notebooks/ablation.ipynb to interpret results ==="
