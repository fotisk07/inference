#!/usr/bin/env bash
# Donut TRAINING quality ablation over BATCH SIZE (at a fixed image size).
#
# For each batch size it fine-tunes a model and writes a field-level F1 summary
# to <OUT_DIR>/batch_<bs>.json. Interpret in notebooks/ablation.ipynb (F1 and
# docs/s vs batch size → the most effective batch for this dataset).
#
# Point at the real dataset and a GPU via the environment:
#   DATA_JSON=/data/train.json DEVICE=cuda bash exp_batch_size.sh
# Defaults run the test_data smoke set. SMOKE=true uses the tiny offline model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."  # run from donut_train/

DATA_JSON="${DATA_JSON:-../test_data/train.json}"
BATCH_SIZES="${BATCH_SIZES:-1,2,4,8}"
IMAGE_SIZE="${IMAGE_SIZE:-1280x960}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
LR="${LR:-3e-4}"
BACKEND="${BACKEND:-sdpa}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
SMOKE="${SMOKE:-false}"
OUT_DIR="${OUT_DIR:-results/ablation}"

H="${IMAGE_SIZE%x*}"
W="${IMAGE_SIZE#*x}"

echo "=== Donut training ablation: batch size ==="
echo "  data_json   = $DATA_JSON"
echo "  batch_sizes = $BATCH_SIZES   (fixed: image=${H}x${W} lr=$LR epochs=$MAX_EPOCHS)"
echo "  backend     = $BACKEND   device=$DEVICE   smoke=$SMOKE"
echo "  output      = $OUT_DIR/batch_<bs>.json"
echo ""

IFS=',' read -ra SIZES <<< "$BATCH_SIZES"
for bs in "${SIZES[@]}"; do
  tag="batch_${bs}"
  echo "--- training batch_size=${bs} → $OUT_DIR/$tag.json ---"
  uv run python train.py \
    --data_json   "$DATA_JSON" \
    --image_size  "[$H,$W]" \
    --batch_size  "$bs" \
    --max_epochs  "$MAX_EPOCHS" \
    --lr          "$LR" \
    --backend     "$BACKEND" \
    --seed        "$SEED" \
    --device      "$DEVICE" \
    --smoke       "$SMOKE" \
    --output_dir  "checkpoints/$tag" \
    --ablation_out "$OUT_DIR/$tag.json"
done

echo ""
echo "=== Done — open notebooks/ablation.ipynb to interpret results ==="
