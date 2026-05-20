#!/usr/bin/env bash
# =============================================================================
# Experiment 5 — Decoder Efficiency: Document Diversity Impact
# =============================================================================
# Goal:
#   Quantify how much GPU compute is wasted on padding when batching diverse
#   real-world documents vs. identical documents.  The decoder runs for
#   max(sequence_lengths) steps for every sample in a batch — shorter
#   sequences sit idle.  This experiment isolates that effect.
#
# Method:
#   uniform — pool_size=1: every batch contains copies of the same image.
#             All output sequences are the same length → efficiency ≈ 1.0.
#             This establishes the theoretical throughput ceiling.
#   diverse — pool_size=0: all 100 cord-v2 test images with variable lengths.
#             Efficiency < 1.0 and degrades as batch size grows.
#
#   Both sweeps: B=1..32, 100 measurement runs, 10 warmup.
#
# Key questions answered:
#   - How much throughput do we lose from padding waste at each batch size?
#   - How variable are the output lengths across real cord-v2 receipts?
#   - Is there a batch size beyond which diversity cost outweighs batching gain?
#
# Outputs  →  logs/exp05_decoder_efficiency/{uniform,diverse}/
#   *_sweep.csv          decoder_efficiency_mean, tokens_per_second_mean per BS
#   *_bs{N}_runs.csv     per-run num_generated_tokens, decoder_efficiency
#
# Visualisations in analysis.ipynb  →  Section 5
#   - Decoder efficiency vs batch size (uniform vs diverse, two lines)
#   - Tokens/sec vs batch size (uniform vs diverse)
#   - Histogram of generated token counts across all images
#
# Estimated runtime:  ~60 min total on a single A100
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Uniform pool (pool_size=1 — same image repeated)"
uv run python run_benchmark.py \
  --device cuda \
  --batch-sizes 1 2 4 8 16 32 \
  --num-runs 100 \
  --warmup-runs 10 \
  --pool-size 1 \
  --output-dir logs/exp05_decoder_efficiency/uniform

echo "==> Diverse pool (pool_size=0 — all 100 test images)"
uv run python run_benchmark.py \
  --device cuda \
  --batch-sizes 1 2 4 8 16 32 \
  --num-runs 100 \
  --warmup-runs 10 \
  --pool-size 0 \
  --output-dir logs/exp05_decoder_efficiency/diverse
