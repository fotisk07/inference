# Metrics — precise definitions

Both donut benches — `bench_infer_step` (one `generate()` call) and
`bench_train_step` (one `forward → backward → optimizer.step`) — plus `train.py`'s
per-epoch line speak **one** metric: **docs/s**. They differ only in *which work
the time window covers*, never in what the number means. This file defines every
variant exactly — its formula, its time window, what is and isn't included, and how
it's measured — so "compute docs/s" means the same thing whether you read it off the
inference table or the training table.

## Vocabulary

- **doc (sample)** — one document = one image (encoder input) + its token sequence.
  In **training** that sequence is the label (decoder target), and teacher-forcing
  pushes the whole sequence through the decoder in a **single** forward pass. In
  **inference** it is the sequence the model generates autoregressively, one token
  per step. Either way, one doc = one image + one sequence.
- **B** — batch size (docs per step).
- **step** — the unit of work timed. Training: one optimizer update over a batch
  (`forward → backward → optimizer.step`). Inference: one `generate()` call over a
  batch (encoder forward + the autoregressive decode loop).
- **Δt** — an elapsed time window, in seconds.

## The one formula

```
docs/s = B / Δt
```

Every "docs/s" number is this. The only thing that differs between them is **which Δt**
— i.e. what work the window covers. Naming the window removes all ambiguity.

## Time windows (the docs/s variants)

| name             | Δt covers                                              | excludes                | where |
|------------------|--------------------------------------------------------|-------------------------|-------|
| **e2e docs/s**   | full step wall-time: data fetch + H2D + fwd + bwd + opt | nothing                 | `train.py` loop |
| **compute docs/s** | the whole step, GPU-synced — train: fwd+bwd+opt; infer: encoder+decode generate | data-fetch wait | `bench_train.py` · `bench_infer_step` · `train.py` loop |
| **encoder docs/s** | encoder forward only                                  | decoder/decode, bwd, opt, data | `bench_train.py` · `bench_infer_step` |
| **val docs/s**   | forward only (loss), GPU-synced, no_grad/eval          | backward, opt, data     | `train.py` `evaluate` |

The same two columns — **compute docs/s** and **encoder docs/s** — appear in both
the inference table (`scripts/inference/bench_speed.py`) and the training table
(`scripts/train/bench_train.py`), computed by the identical formula `B / Δt`. The
only thing that changed across them is the **step**: a `generate()` call vs a
fwd+bwd+opt update.

- **e2e** is the *practical* throughput — what a real run actually achieves, dataloader
  and all. This is the number that decides if training is faster in the real world.
- **compute** is the *hardware-limited* throughput — what the GPU does once data is in
  hand. The accel backends can only ever change this one.
- **encoder** isolates the encoder forward, where the Swin SDPA patch lives, so its
  effect is visible separately from the (constant) decoder.
- **val** is forward-only, so per-doc it is faster than a train step (no backward, no
  optimizer); reported to confirm and quantify that gap.

> "docs/s" unqualified = the whole step (e2e in the real loop, compute in the bench).
> Anything encoder-only is labelled `encoder_*`.

## Component breakdown (ms per step)

`bench_train.py` times four nested regions separately (each its own warmup + repeats),
then derives the components by subtraction:

| measured region | what runs |
|-----------------|-----------|
| `encoder_fwd`   | `model.encoder(pixel_values)`, no_grad |
| `forward`       | full `model(pixel_values, labels).loss`, no_grad |
| `forward+backward` | forward, then `loss.backward()` |
| `full_step`     | forward, backward, `optimizer.step()` |

Derived components (what the table prints):

```
encoder_fwd  = encoder_fwd
decoder_fwd  = forward            − encoder_fwd
backward     = (forward+backward) − forward
optim_step   = full_step          − (forward+backward)
total        = full_step
```

These sum (by construction) to `total`. Because backward/optim are **differences of two
separately-timed means**, they are noisier than the directly-timed regions and are
clamped at 0 — on a fast/tiny/CPU run a derived component can read `0.00` from timing
noise. Trust the directly-measured `encoder_fwd`, `forward`, and `total` most; read the
derived ones as approximate attribution.

`bench_infer_step` uses the **same** two-region / subtraction recipe, just simpler —
it directly times `encoder_fwd` and the full `generate()` (`total`), then derives:

```
encoder_fwd = encoder_fwd
decode      = total − encoder_fwd
total       = full generate()
```

so `encoder_fwd_ms` / `decode_ms` / `total_ms` and the `compute_docs_s` /
`encoder_docs_s` columns line up one-for-one with the training table.

## Wall-clock split: data % / compute % / overhead %

The epoch wall (`epoch_secs`, the **e2e** window) splits **three** ways — all measured
in the `train.py` loop, all over `epoch_secs`, summing to 100 %:

```
data %     = data_fetch                       / epoch_secs × 100
compute %  = compute                          / epoch_secs × 100
overhead % = (epoch_secs − data_fetch − compute) / epoch_secs × 100
```

- `data_fetch` — wall-time *waiting for the next batch* from the DataLoader.
- `compute` — the synced H2D+fwd+bwd+opt region.
- **overhead** — everything else the wall clock contains but neither bucket times:
  per-step `loss.item()`, the tqdm postfix, per-step mlflow logging, the step sync.

Because `compute` is a **sub-interval** of the epoch wall, `compute_docs_s ≥ e2e_docs_s`
**always**, and the two are tied by an exact identity:

```
compute % = e2e_docs_s / compute_docs_s × 100
```

So the gap between e2e and compute throughput is exactly `data % + overhead %`. A low
`data %` with a large e2e↔compute gap is not a contradiction — the gap is **overhead**,
not the dataloader. Use the three buckets to attribute it:

- high **data %** → dataloader-bound: **e2e docs/s stays flat across backends even when
  compute docs/s improves** (a kernel optimization cannot fix it).
- high **overhead %** → per-step Python/logging/sync cost dominates the wall.
- both low → e2e ≈ compute: the GPU step is the wall, and accel backends move it.

`bench_loader.py` measures a standalone **loader docs/s** (real images through
`DonutDataset`); comparing it to compute docs/s shows the data side from the bench.

## How it's measured

- **Timing harness:** `donut.bench.time_fn` — `n_warmup` discarded iterations, then
  `n_runs` timed, each bracketed by `torch.cuda.synchronize()` so async GPU work is
  finished before the clock is read. Reports `mean / std / p50 / p95` ms; the docs/s
  numbers use the mean.
- **Sync:** on CPU, `synchronize` is a no-op. The per-step sync adds a small fixed
  overhead — fine for **comparing** backends, but don't read absolute ms as
  zero-overhead.
- **Numerics:** the bench mirrors real training — fp32 master weights +
  `torch.autocast(bf16)` (the `--precision` knob), so the accel kernels run in the same
  dtype they would during training.
- **Peak memory:** `donut.bench._peak_mem_mb` resets the CUDA peak counter, runs one
  `full_step`, and reads `max_memory_allocated` (MB). `None` on CPU.
- **Isolation:** `bench_train.py` reuses **one** fixed in-memory batch across all runs,
  so the only thing varying between backends is the kernel — no dataloader/disk noise.

## Reading the result

The accel "speeds up training" iff a backend lowers the **total** step time (raises
**compute docs/s**) versus `baseline`. To locate *where* a change comes from, compare
**encoder docs/s** (the Swin patch is the only encoder difference; the decoder is the
same SDPA path in every backend including baseline — the bench prints each component's
attn impl to confirm). Whether that compute win shows up in real training is answered by
**e2e docs/s** and the **data % / overhead %** split: a high data % means the win is
hidden behind the dataloader; a high overhead % means it's hidden behind per-step
Python/logging cost.
