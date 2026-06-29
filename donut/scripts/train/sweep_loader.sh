#!/usr/bin/env bash
# Sweep the dataloader benchmark over workers × image-size × batch.
# The CLI sweeps internally over the comma-lists, so this is a single call.
# Override the dataset with: DATA_JSON=/path/to/train.json ./sweep_loader.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # → donut/

DATA_JSON="${DATA_JSON:-../test_data/train.json}"

uv run python scripts/train/bench_loader.py "$DATA_JSON" \
  --num-workers 0,2,4,8,16 \
  --image-sizes 1280x960,1920x1440 \
  --batch-sizes 1,4,8
