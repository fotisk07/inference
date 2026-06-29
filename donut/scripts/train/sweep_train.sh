#!/usr/bin/env bash
# Sweep training over backend × image-size × batch. train.py runs ONE config per
# invocation, so loop the grid here; each run gets a unique --run-name so checkpoints
# and metric records don't collide.
# Override: DATA_JSON=/path/train.json EPOCHS=10 ./sweep_train.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # → donut/

DATA_JSON="${DATA_JSON:-../test_data/train.json}"
EPOCHS="${EPOCHS:-30}"

BACKENDS=(baseline eager sdpa fa)   # fa needs the flash-attn-4 extra — trim if absent
SIZES=("1280 960" "1920 1440")      # "height width"
BATCHES=(1 4 8)

for b in "${BACKENDS[@]}"; do
  for s in "${SIZES[@]}"; do
    set -- $s; H=$1; W=$2
    for bs in "${BATCHES[@]}"; do
      run="bk${b}_${H}x${W}_bs${bs}"
      echo ">>> $run"
      uv run python scripts/train/train.py \
        --data-json "$DATA_JSON" --backend "$b" \
        --image-height "$H" --image-width "$W" \
        --batch-size "$bs" --max-epochs "$EPOCHS" \
        --run-name "$run" --seed 42
    done
  done
done
