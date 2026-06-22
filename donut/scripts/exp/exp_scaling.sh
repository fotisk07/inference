#!/usr/bin/env bash
# Experiment 1 -- the main grid: how the acceleration ladder scales with image
# size and batch size. 3 resolutions x 4 batch sizes x 8 presets, fixed
# max_new_tokens=128. Carries the baseline->eager->sdpa/fa ladder, the
# image/batch scaling curves, the auto-sdpa-backend bracket (sdpa_{flash,math,
# efficient,cudnn}), and peak GPU memory. Resumable: re-running skips configs
# whose per-config JSON already exists (pass --force to recompute).
#
# Usage: bash scripts/exp/exp_scaling.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

BACKENDS="baseline,eager,sdpa,sdpa_flash,sdpa_math,sdpa_efficient,sdpa_cudnn,fa"

uv run python scripts/bench_speed.py \
  --backends "$BACKENDS" \
  --image-sizes 1280x960,1920x1440,2560x1920 \
  --batch-sizes 1,2,4,8 \
  --max-new-tokens 128 \
  --out results/exp_scaling \
  "$@"
