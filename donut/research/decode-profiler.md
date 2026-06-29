# Research brief — Decode profiler (confirm the bound before you optimize)

**Branch:** `research/decode-profiler`
**Boilerplate:** [`donut/scripts/profiling/profile_decode.py`](../scripts/profiling/profile_decode.py)
**Status:** skeleton runs + dumps a trace; the analysis is yours to write tomorrow.

## Why this exists (read first)

You already proved with `bench_speed.py` that decode is the expensive part of a
donut forward. What you have NOT proven is **why**. The whole compile / static-cache /
quant program rests on one assumption:

> At `query_len=1`, autoregressive decode is **launch-overhead + memory-bandwidth
> bound, not compute bound**.

If that's true, `torch.compile` + CUDA graphs + a static KV cache (Branch
`research/compile-static-cache`) is the highest-ROI lever, because it removes
per-token Python/launch overhead. If it's *false* (decode is actually GEMM-bound),
compile buys little and you should jump straight to quantization instead. **This
branch is the cheap experiment that tells you which branch to fund.** Do it first.

## Hypothesis

Per decode step the decoder runs one tiny `q=1` attention + a few small GEMMs over
cached KV. Each step is dozens of micro-kernels, each finishing faster than the CPU
can queue the next → the GPU idles in **launch gaps**. Expectation:

- summed CUDA kernel time ≪ wall time for the decode region (large launch-gap fraction);
- the dominant kernels are small GEMMs / elementwise / memcpy, all **bandwidth-bound**
  (low arithmetic intensity), not big compute-bound matmuls;
- the **encoder** (Branch context) is the opposite — compute-bound, amortizes over batch.

## What the skeleton already does

- Loads model, applies a backend preset, warms up once (skip first-call autotune).
- Wraps **one `generate()`** and **one train step** in `torch.profiler` (CPU+CUDA).
- Exports a chrome trace (`results/profile_decode/*.json`) → open in Perfetto /
  `chrome://tracing`.
- Prints `key_averages()` top-25 by CUDA time.

Run:
```
uv run donut/scripts/profiling/profile_decode.py --backend sdpa
uv run donut/scripts/profiling/profile_decode.py --backend sdpa --batch-size 8
uv run donut/scripts/profiling/profile_decode.py --backend baseline --skip-train
```

## What you flesh out tomorrow (the TODOs in `_summarize`)

1. **Launch-gap fraction.** `total_cuda_time / wall_time` for the decode region.
   Low ratio ⇒ launch-bound ⇒ compile/CUDA-graphs wins. This is THE number.
2. **Kernel buckets.** Group `key_averages()` rows by name into {gemm, attention,
   elementwise, memcpy/cast, other}. Where does decode time actually go?
3. **Bandwidth roofline (decode).** For the decoder's per-step GEMMs (shapes
   `q=1, kv=t`), compute bytes moved vs achieved GB/s; compare to device peak.
   Confirms bandwidth-bound.
4. **Batch sensitivity.** Re-run at bs ∈ {1,4,8,16}. Launch-gap fraction should
   *shrink* as batch grows (more work per launch) — quantifies how much batching
   alone already hides the overhead, and at what batch compile stops mattering.
5. **Inference vs training contrast.** Same buckets on the train step — expect it
   compute-bound (validates that Branch `research/compile-training` is a *different*
   kind of win than the inference branch).

## Decision rule (the deliverable)

| Finding | Action |
|---|---|
| Decode launch-gap fraction high (e.g. >40% idle), kernels bandwidth-bound | **Fund `compile-static-cache` first.** Expect CUDA graphs to recover most of the gap. |
| Launch-gap small, decode GEMM-bound | Compile won't help much → **skip to quantization** (int8/fp8 decoder). |
| Gap collapses by bs≈8 | Cheapest win is just **bigger inference batch**; compile only for low-latency single-doc. |
| Train step compute-bound (expected) | `compile-training` ROI = kernel fusion, independent of the above. |

## Notes / gotchas
- Profile **steady-state**, not the first call — skeleton warms up once; keep that.
- `record_shapes=True` is on so you can read GEMM shapes for the roofline.
- `min_new_tokens == max_new_tokens` (matches `bench_infer_step`) so every run decodes
  a fixed length — comparable across backends.
- Don't trust `key_averages()` wall alone for the gap; cross-check against the trace
  timeline (gaps between CUDA kernels on the stream row).
