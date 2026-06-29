# Donut

OCR-free document understanding: feed in an image of a document, get back
structured text. This wraps HuggingFace's pretrained
[Donut](https://huggingface.co/naver-clova-ix/donut-base) (`naver-clova-ix/donut-base`)
behind a single `load_model()` call with toggleable attention backends.

## Install

```bash
uv sync
```

## Load it

One line. Device, dtype, and acceleration backend are picked for you:

```python
from donut import load_model

model, processor = load_model()  # cuda+bf16 if available, else cpu+fp32; sdpa backend
```

Weights download from the HuggingFace Hub on first use.

## Run inference

Standard HuggingFace `generate()` — nothing custom to learn:

```python
from PIL import Image

image = Image.open("document.png")
inputs = processor(images=image, return_tensors="pt").to(model.device)

ids = model.generate(**inputs, max_new_tokens=128)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

## Pick a backend

`load_model(backend=...)` swaps the attention implementation. All presets are
numerically equivalent; they trade speed for portability:

| Backend          | What it does                                  |
| ---------------- | --------------------------------------------- |
| `baseline`       | Stock HF, no changes                          |
| `eager`          | Mask caching only                             |
| `sdpa` (default) | Mask caching + PyTorch fused attention        |
| `sdpa_flash`     | SDPA pinned to the flash kernel               |
| `sdpa_cudnn`     | SDPA pinned to the cuDNN kernel               |
| `sdpa_math`      | SDPA pinned to the math kernel                |
| `sdpa_efficient` | SDPA pinned to the memory-efficient kernel    |
| `fa`             | FlashAttention-4 (needs the `flash-attn-4` extra) |

```python
model, processor = load_model(backend="fa")
```

Backends apply in-place and are reversible:

```python
from donut import apply_accel, revert_accel

revert_accel(model)            # back to stock eager
apply_accel(model, "sdpa")     # re-apply (idempotent)
```

## Going deeper

See [`docs/attention-backends.md`](docs/attention-backends.md) and
[`notebooks/story.ipynb`](notebooks/story.ipynb) for the optimization details
and benchmarks.

## Fine-tuning

Where everything above answers "how *fast* is the model," fine-tuning answers
"how *accurate* is it." The dataset (`donut.dataset`) and metrics
(`donut.metrics`) live in `src/`; the training loop itself lives in its CLI,
`scripts/train/train.py`, which leans on small model-config helpers in
`donut.model` (`set_shift_tokens`, `fit_decoder_to_vocab`, `autocast`). Training
runs on top of `load_model`, so the chosen accel backend (default `sdpa`) is
active during training too.

```bash
uv sync --extra train          # adds MLflow; base `uv sync` already covers predict
```

**Data** is one aggregate JSON — each record is a document (image path relative
to the JSON, plus its ground-truth fields):

```json
[{"image": "images/train/doc_01.jpg",
  "fields": [{"field_name": "BR/COMMISSION/E-mail", "annotator_text": "a@b.com"}]}]
```

Only the last `/`-segment of `field_name` is used; the field vocabulary lives in
[`src/donut/constants.py`](src/donut/constants.py) (`TASK_TOKEN`, `FIELD_TOKENS`).
Per-image annotations can be folded into this format with
`scripts/train/migrate_to_aggregate_json.py`.

```bash
# 1. Smoke test — tiny offline model on CPU, seconds. Zero exit = OK.
uv run python scripts/train/train.py --smoke

# 2. Real fine-tune (downloads donut-base the first time).
uv run python scripts/train/train.py --data-json /path/train.json --device cuda \
  --image-height 1280 --image-width 960 --batch-size 4 --max-epochs 30 --seed 42

# 3. Score a saved checkpoint (prints strict + soft P/R/F1 tables).
uv run python scripts/inference/predict.py checkpoints/best --data-json /path/val.json
```

Checkpoints are HuggingFace `save_pretrained` dirs — `best/` (lowest val loss)
and `last/` — each holding the model and the processor (with the registered field
tokens and image size), so `predict.py` rebuilds the exact model from the dir.
`train.py --help` / `predict.py --help` list every flag.

### Quality

Every `(document, field)` pair is scored TP / FP / FN / TN against ground truth
(a *wrong* value is an FN, not an FP; an FP is predicting a field that has
nothing to predict). From these: **precision**, **recall**, **F1**, reported in
**strict** (exact) and **soft** (lowercase + trimmed) modes, plus per-document
buckets (`perfect`, `fn_only`, `fp_only`, `mixed`).

### Does accel speed up *training*?

A separate question from inference. Two measurements, see the
[Metrics](#metrics) section below for exact definitions:

```bash
# Mechanism: per-backend training-step breakdown, dataloader stripped.
uv run python scripts/train/bench_train.py --backends baseline,eager,sdpa,fa
#   --image-sizes 1280x960,1920x1440 --batch-sizes 1,4   sweep size/batch (like bench_speed)
#   add --tiny             to run the harness on CPU with no downloads

# Dataloader: standalone real-data loading throughput (the bottleneck bench_train strips).
uv run python scripts/train/bench_loader.py /path/to/data.json --num-workers 0,4,8

# End-to-end: train.py prints e2e / compute docs/s + data % / overhead % per epoch.
```

The atomic timer is `donut.bench.bench_train_step` (twin of the inference
`bench_infer_step`, same docs/s metric);
if compute got faster but `data %` is high, the dataloader — not the kernel —
is the wall-clock lever (a high `overhead %` instead points at per-step Python cost).

## Metrics

Both benches — `bench_infer_step` (one `generate()` call) and `bench_train_step`
(one `forward → backward → grad-clip → optimizer.step`) — plus `train.py`'s
per-epoch line speak **one** metric: **docs/s**. They differ only in *which work the
time window covers*, never in what the number means. This section defines every
variant exactly so "compute docs/s" means the same thing whether you read it off the
inference table or the training table.

### TL;DR

```
docs/s = B / Δt          B = batch size (docs/step), Δt = an elapsed window (s)
```

Every docs/s number is this one formula. Only the **window** changes:

| metric             | Δt covers                                                       | reported by |
|--------------------|-----------------------------------------------------------------|-------------|
| **e2e docs/s**     | full step wall: data fetch + H2D + fwd + bwd + clip + opt       | `train.py` loop |
| **compute docs/s** | the GPU step, synced — train & bench: `fwd+bwd+clip+opt` · infer: `encoder+decode` | `train.py` · `bench_train.py` · `bench_speed.py` |
| **encoder docs/s** | encoder forward only                                            | `bench_train.py` · `bench_speed.py` |
| **val docs/s**     | forward only (loss), synced, no_grad/eval                       | `train.py` `evaluate` |
| **loader docs/s**  | wall to produce one ready batch from real data                  | `bench_loader.py` |

All timers use `time.perf_counter` (monotonic, hi-res); every GPU window is bracketed
by `torch.cuda.synchronize()` so async work is finished before the clock is read.

- **e2e** is the *practical* throughput — what a real run achieves, dataloader and all.
- **compute** is the *hardware-limited* throughput — what the GPU does once data is in
  hand. The accel backends can only ever move this one. **The train and bench compute
  windows hold the identical op set** (`fwd+bwd+clip+opt`, tensors already on device),
  so the two numbers are directly comparable.
- **encoder** isolates the Swin SDPA patch, separate from the (constant) decoder.
- **val** is forward-only, so per-doc faster than a train step; quantifies that gap.

### Vocabulary

- **doc (sample)** — one document = one image (encoder input) + its token sequence.
  In **training** that sequence is the label and teacher-forcing pushes it through the
  decoder in a **single** forward pass; in **inference** it is generated
  autoregressively, one token per step. Either way: one doc = one image + one sequence.
- **B** — batch size (docs per step).
- **step** — the unit of work timed. Training: one optimizer update over a batch
  (`fwd → bwd → clip → opt`). Inference: one `generate()` call (encoder forward + the
  autoregressive decode loop).
- **Δt** — an elapsed time window, in seconds.

> "docs/s" unqualified = the whole step (e2e in the real loop, compute in the bench).
> Anything encoder-only is labelled `encoder_*`.

### Component breakdown (ms per step)

`bench_train.py` times four nested regions separately (each its own warmup + repeats),
then derives the components by subtraction:

| measured region    | what runs |
|--------------------|-----------|
| `encoder_fwd`      | `model.encoder(pixel_values)`, no_grad |
| `forward`          | full `model(pixel_values, labels).loss`, no_grad |
| `forward+backward` | forward, then `loss.backward()` |
| `full_step`        | forward, backward, grad-clip, `optimizer.step()` |

```
encoder_fwd  = encoder_fwd
decoder_fwd  = forward            − encoder_fwd
backward     = (forward+backward) − forward
optim_step   = full_step          − (forward+backward)   # = grad-clip + optimizer.step
total        = full_step
```

These sum (by construction) to `total`. Because backward/optim are **differences of two
separately-timed means**, they are noisier than the directly-timed regions and are
clamped at 0 — on a fast/tiny/CPU run a derived component can read `0.00` from timing
noise. Trust `encoder_fwd`, `forward`, and `total` most; read the derived ones as
approximate attribution.

`bench_infer_step` uses the **same** two-region / subtraction recipe, just simpler — it
directly times `encoder_fwd` and the full `generate()` (`total`), then derives
`decode = total − encoder_fwd`, so its `encoder_fwd_ms` / `decode_ms` / `total_ms` and the
`compute_docs_s` / `encoder_docs_s` columns line up one-for-one with the training table.

### Wall-clock split: data % / compute % / overhead %

The epoch wall (`wall_s`, the **e2e** window) splits **three** ways — all measured in the
`train.py` loop, all over `wall_s`, summing to 100 %:

```
data %     = data_fetch                  / wall × 100
compute %  = compute                     / wall × 100
overhead % = (wall − data_fetch − compute) / wall × 100
```

- `data_fetch` — wall waiting on the next batch from the DataLoader.
- `compute` — the synced `fwd+bwd+clip+opt` region (the op set the bench measures).
- **overhead** — everything else the wall contains: the **H2D copy** (`.to(device)`),
  `scheduler.step()`, per-step `loss.item()`, the tqdm postfix, mlflow logging, the sync.
  (H2D and scheduler are deliberately *outside* compute so `compute docs/s` is
  apples-to-apples with the bench, where tensors are already resident and there is no
  scheduler.)

Because `compute` is a **sub-interval** of the wall, `compute_docs_s ≥ e2e_docs_s`
**always**, tied by the exact identity `compute % = e2e_docs_s / compute_docs_s × 100`.
So the e2e↔compute gap is exactly `data % + overhead %` — not the dataloader alone:

- high **data %** → dataloader-bound: e2e docs/s stays flat across backends even when
  compute docs/s improves (a kernel can't fix it).
- high **overhead %** → per-step Python/logging/sync/H2D cost dominates the wall.
- both low → e2e ≈ compute: the GPU step is the wall, and accel backends move it.

`bench_loader.py` measures a standalone **loader docs/s** (real images through
`DonutDataset`); comparing it to compute docs/s shows the data side from the bench.

### How it's measured

- **Timing harness:** `donut.bench.time_fn` — `n_warmup` discarded iterations, then
  `n_runs` timed, each bracketed by `torch.cuda.synchronize()`. Reports
  `mean / std / p50 / p95` ms; the docs/s numbers use the mean. `train.py` times its
  regions inline with the same `perf_counter` + sync recipe.
- **Sync:** on CPU `synchronize` is a no-op. The per-step sync adds a small fixed
  overhead — fine for *comparing* backends, but absolute ms is not zero-overhead.
- **Numerics:** the bench mirrors real training — fp32 master weights +
  `torch.autocast(bf16)` (the `--precision` knob), so accel kernels run in the training dtype.
- **Peak memory:** `donut.bench._peak_mem_mb` resets the CUDA peak counter, runs one
  `full_step`, reads `max_memory_allocated` (MB). `None` on CPU.
- **Isolation:** `bench_train.py` reuses **one** fixed in-memory batch across all runs, so
  the only thing varying between backends is the kernel — no dataloader/disk noise.

### Reading the result

The accel "speeds up training" iff a backend lowers the **total** step time (raises
**compute docs/s**) versus `baseline`. To locate *where* a change comes from, compare
**encoder docs/s** (the Swin patch is the only encoder difference; the decoder is the
same SDPA path in every backend including baseline). Whether that compute win shows up in
real training is answered by **e2e docs/s** and the **data % / overhead %** split: a high
data % means the win is hidden behind the dataloader; a high overhead % means it's hidden
behind per-step Python/logging/H2D cost.
