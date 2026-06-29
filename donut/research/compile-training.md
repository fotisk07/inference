# Research brief — torch.compile the training step

**Branch:** `research/compile-training`
**Boilerplate:** `compile_model()` in [`donut/src/donut/model.py`](../src/donut/model.py)
(+ commented hook in [`scripts/train/bench_train.py`](../scripts/train/bench_train.py))
**Independent of** the inference branches — this is a *different* kind of win.

## Why this exists

Inference decode is launch-bound (q=1). **Training is the opposite: compute-bound.**
Teacher-forced training runs the full sequence in one forward (large `query_len`), so the
step is dominated by real GEMM/attention FLOPs in fwd + bwd, not by per-token launch
overhead. The lever there is **kernel fusion**: `torch.compile` fuses elementwise +
normalization + attention epilogues, cutting memory traffic and kernel count in the
fwd/bwd. Realistic target: **~1.1–1.4×** step throughput, no math change.

## Hypothesis

Compiling the model improves `compute_docs_s` in `bench_train_step` by fusing fwd/bwd
kernels, with the gain concentrated in `decoder_fwd_ms` + `backward_ms` (the compute-heavy
components) and little change to `optim_ms`. Loss trajectory stays equivalent to eager over
a few steps (compile is semantics-preserving).

## Where to compile (replace the TODO in `compile_model()`)

- `return torch.compile(model, mode="default")` first; then try `"max-autotune"` (longer
  compile, sometimes faster steady-state).
- Wire it in **two call sites**:
  - `build_model()` in [`train.py`](../scripts/train/train.py) — the real run (add a
    `--compile` flag on the `Config`).
  - `bench_train.py` at the marked hook — to measure the breakdown (add `--compile` + a
    "compiled" column).
- Compile the **whole** `VisionEncoderDecoderModel`, not just the decoder: in training the
  encoder forward is a real cost (unlike inference where it amortizes), so it benefits from
  fusion too.

## How to measure (reuse existing infra)

- **Speed/breakdown:** `scripts/train/bench_train.py --backends sdpa --compile` vs without.
  Read the component table (`encoder_fwd_ms`, `decoder_fwd_ms`, `backward_ms`, `optim_ms`,
  `compute_docs_s`). The bench already discards `n_warmup` steps → first-step compile latency
  is excluded from steady-state, which is what we want.
- **Compile latency, reported separately:** time the very first step explicitly; it can be
  tens of seconds. Matters for short runs and for the `train.py` epoch-0 timing.
- **Loss parity:** run a handful of `train.py` steps compiled vs eager from the same seed;
  losses should track (small numeric drift from fusion order is OK, divergence is not).
- **End-to-end:** confirm `train.py` epoch `compute_docs_s` improves and the data loader
  (`bench_loader.py`) isn't the actual bottleneck — compile only helps if compute is the
  limiter (cross-check loader docs/s vs compute docs/s, per the e2e/compute split train.py
  already reports).

## Risks / gotchas
- **Graph breaks** under `autocast` + `clip_grad_norm_` + the HF loss path silently disable
  fusion. Surface with `TORCH_LOGS=graph_breaks` / `fullgraph=True`, fix, then relax.
- **Recompiles** when `batch_size` / `max_length` vary across the sweep — mark dynamic or fix
  shapes per run.
- **AdamW**: try `fused=True` independently; it's a cheap, orthogonal `optim_ms` win and a
  useful baseline to attribute how much of the gain is compile vs optimizer.
- **bf16 autocast interaction**: keep `autocast()` as the single precision source (model.py);
  compile must wrap it, not replace it.

## Definition of done
- `bench_train.py --compile` shows higher `compute_docs_s` vs eager, gain localized to
  fwd/bwd, with first-step compile latency reported separately.
- A few `train.py` steps show loss parity compiled vs eager.
- Brief updated with measured speedup + which components moved.
