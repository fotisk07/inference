# Research brief — torch.compile + static KV cache (inference decode)

**Branch:** `research/compile-static-cache`
**Boilerplate:** [`donut/src/donut/accel/decoder_compiled.py`](../src/donut/accel/decoder_compiled.py)
(+ commented registry hook in [`accel/__init__.py`](../src/donut/accel/__init__.py))
**Depends on:** the verdict from `research/decode-profiler`. If decode is NOT launch-bound,
deprioritize this and go to quantization instead.

## Why this exists

Decode runs at `query_len=1`: per token the decoder does one tiny attention + a handful of
small GEMMs over cached KV, then Python loops to the next token. The GPU finishes each
micro-kernel faster than the CPU can queue the next → **launch-gap idle**. Two changes kill
that overhead together:

1. **Static KV cache** — every decode step gets identical tensor shapes (cache pre-allocated
   to max length, not grown). Stable shapes are the precondition for graph capture.
2. **`torch.compile(mode="reduce-overhead")`** — captures CUDA graphs and replays the whole
   per-step kernel sequence with one launch instead of dozens.

This is the single highest-ROI inference lever the repo doesn't have. Everything else
(attention kernels) is already done.

## Hypothesis

On a launch-bound decode, compile + static cache + CUDA graphs recovers most of the
launch-gap fraction measured in `research/decode-profiler`. Target: meaningfully higher
`compute_docs_s` / lower `decode_ms` at bs=1 (where overhead dominates), shrinking as batch
grows. Greedy output stays **bit-identical** (no math change) → audit must be exact-match.

## Exact knobs to wire (replace the TODOs in `decoder_compiled.py`)

- `model.generation_config.cache_implementation = "static"` (already in the stub).
- `model.decoder.forward = torch.compile(model.decoder.forward, mode="reduce-overhead",
  fullgraph=False)`. Start `fullgraph=False`, then tighten to `True` once graph breaks are
  gone (a break silently disables the CUDA-graph win).
- Decide **what** to compile: just `model.decoder.forward`, or `model.forward` so the
  cross-attention read of the encoder KV is captured too. Measure both.
- Mark dynamic dims or pad to buckets so changing `batch_size` / `kv_len` doesn't trigger a
  recompile mid-sweep.

## How to measure (reuse existing infra)

- **Speed:** add the `sdpa_compiled` preset (uncomment the registry hook), then run
  `scripts/inference/bench_speed.py --backends sdpa,sdpa_compiled`. Compare `decode_ms`,
  `total_ms`, `compute_docs_s`, `peak_mem_mb`. Sweep `--batch-sizes 1,4,8,16` to see where the
  win fades.
- **Warmup accounting:** `bench_infer_step` already discards `n_warmup` runs — good, because
  the first compiled call pays compile + graph-capture cost. **Report compile latency
  separately** (time the first call) so it's not hidden; it matters for short-lived processes.
- **Correctness:** `scripts/inference/audit_accel.py` — `sdpa_compiled` must match the eager
  baseline exactly (max-AE ≈ 0, exact sequence match). Static cache changes shapes, not math.

## Risks / gotchas (the real work is here)

- **Recompiles** on shape change blow away the win — watch `TORCH_LOGS=recompiles`.
- **Graph breaks** under `generate()`'s sampling/stopping logic — `fullgraph=True` to surface
  them, fix, then relax. The VisionEncoderDecoder `generate` wrapper is the likely break site.
- **Static cache + `min_new_tokens==max_new_tokens`** (bench forces fixed length) interact:
  make sure the cache is sized to `max_new_tokens` and not re-allocated per call.
- **cuDNN/flash kv_len=1** quirk (see `decoder_sdpa.py`) still applies under compile — keep the
  same backend fallbacks; compile sits on top of the attention backend, doesn't replace it.
- **First-call cost** can dwarf the per-step saving for tiny `max_new_tokens` — quantify the
  break-even token count.

## Definition of done
- `sdpa_compiled` preset passes `audit_accel.py` exact-match.
- `bench_speed.py` shows `decode_ms` reduction vs `sdpa` at bs=1, with compile latency reported.
- Brief updated with the measured launch-gap recovery vs the profiler's predicted ceiling.
