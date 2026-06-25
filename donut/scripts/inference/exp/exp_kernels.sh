#!/usr/bin/env bash
# Experiment 3 -- kernel isolation: the real FA4 kernel vs each SDPA backend on
# synthetic q/k/v at the real decoder shape (16 heads, head_dim 64, bf16),
# decoupled from the model and HF overhead. decode mode = query_len 1 (the real
# per-step shape); prefill mode = query_len == kv_len, causal (FA's favorable
# regime). This is the root-cause evidence: it shows the FA4 CUTLASS kernel is
# itself slower than cudnn/efficient at the tiny decode shape -- it's the
# kernel, not HF overhead.
#
# Usage: bash scripts/exp/exp_kernels.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

uv run python scripts/inference/bench_attention_kernels.py \
  --kv-lens 1,8,32,128,512,2048,4096 \
  --batch-sizes 1,8,32 \
  --modes decode,prefill \
  --out results/exp_kernels \
  "$@"
