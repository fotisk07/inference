#!/usr/bin/env bash
# Donut TRAINING quality ablation over IMAGE SIZE — the central axis.
#
# For each HxW it fine-tunes a model and writes a field-level F1 summary to
# <OUT_DIR>/imgsize_<H>x<W>.json. Interpret in notebooks/ablation.ipynb (F1 vs
# image size → smallest resolution that still reads the document) and join with
# the inference speed numbers in notebooks/pareto.ipynb.
#
# Use the SAME HxW values here as in donut/scripts/run_ablation.sh so the Pareto
# join on image_size lines up.
#
# Point at the real dataset and a GPU via the environment:
#   DATA_JSON=/data/train.json DEVICE=cuda bash exp_image_size.sh
# Defaults run the test_data smoke set. SMOKE=true uses the tiny offline model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."  # run from donut_train/

DATA_JSON="${DATA_JSON:-../test_data/train.json}"
IMAGE_SIZES="${IMAGE_SIZES:-640x480,960x720,1280x960}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
LR="${LR:-3e-4}"
BACKEND="${BACKEND:-sdpa}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
SMOKE="${SMOKE:-false}"
OUT_DIR="${OUT_DIR:-results/ablation}"

echo "=== Donut training ablation: image size ==="
echo "  data_json   = $DATA_JSON"
echo "  image_sizes = $IMAGE_SIZES   (fixed: bs=$BATCH_SIZE lr=$LR epochs=$MAX_EPOCHS)"
echo "  backend     = $BACKEND   device=$DEVICE   smoke=$SMOKE"
echo "  output      = $OUT_DIR/imgsize_<HxW>.json"
echo ""

IFS=',' read -ra SIZES <<< "$IMAGE_SIZES"
for size in "${SIZES[@]}"; do
  H="${size%x*}"
  W="${size#*x}"
  tag="imgsize_${H}x${W}"
  echo "--- training image_size=${H}x${W} → $OUT_DIR/$tag.json ---"
  uv run python train.py \
    --data_json   "$DATA_JSON" \
    --image_size  "[$H,$W]" \
    --batch_size  "$BATCH_SIZE" \
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
