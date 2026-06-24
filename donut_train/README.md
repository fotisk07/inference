# `donut_train` — fine-tuning & quality scoring

Fine-tune Donut for **field extraction** and measure how good it is. Where the
`donut` package answers "how *fast* is the model," this one answers "how
*accurate* is it." Training runs on top of `donut.load_model`, so the chosen
acceleration backend (default `sdpa`) is active during training too.

---

## Data format

One aggregate JSON file. Each record is a document: an image path (relative to
the JSON's folder) plus its ground-truth fields.

```json
[
  {
    "image": "images/train/doc_01.jpg",
    "fields": [
      {"field_name": "BR/COMMISSION/E-mail",     "annotator_text": "a@b.com"},
      {"field_name": "BR/COMMISSION/data_emissao", "annotator_text": "15/03/2024"}
    ]
  }
]
```

Only the last `/`-segment of `field_name` is used as the field name. The set of
valid fields lives in `config.yaml` (`TASK_TOKEN`, `FIELD_TOKENS`).

---

## Quick start

The CLIs use [Typer](https://typer.dev): flags are hyphenated (`--data-json`),
booleans are switches (`--smoke`, `--no-token2json-format`), and
`python train.py --help` / `python predict.py --help` list everything.

```bash
# 1. Smoke test — tiny offline model on CPU, seconds. Proves the
#    pipeline works end-to-end. Zero exit = OK.
uv run python train.py --smoke

# 2. Real fine-tune (downloads donut-base the first time).
uv run python train.py --data-json /path/train.json --device cuda \
  --image-height 1280 --image-width 960 --batch-size 4 --max-epochs 30 --seed 42

# 3. Score a saved checkpoint on labelled data (prints P/R/F1 tables).
#    The checkpoint is a saved dir: checkpoints/best (lowest val loss) or
#    checkpoints/last (final epoch). Add --output-json preds.json to dump
#    per-document {image, gt, pred} records.
uv run python predict.py checkpoints/best --data-json /path/val.json
```

---

## `train.py` — the knobs that matter

| flag                      | default        | meaning                                                        |
|---------------------------|----------------|----------------------------------------------------------------|
| `--data-json`             | test_data      | aggregate JSON described above.                               |
| `--val-split`             | 0.2            | fraction held out for validation.                            |
| `--image-height/-width`   | 1280 / 960     | `(H,W)` fed to the encoder. **Lower = faster + less VRAM, but less legible.** |
| `--batch-size`            | 4              | docs per step. Higher = faster/epoch but more VRAM.          |
| `--lr`                    | 3e-4           | AdamW learning rate.                                         |
| `--max-epochs`            | 30             | training epochs.                                            |
| `--max-length`            | 128            | max decoder tokens (also the eval generation cap).          |
| `--backend`               | sdpa           | attention backend from `donut`, active in training (eager/sdpa/fa). |
| `--precision`             | bf16           | on CUDA: fp32 master weights + bf16 autocast compute. `fp32` for full fp32. |
| `--seed`                  | None           | set it (e.g. 42) for reproducible shuffles/splits.          |
| `--weight-decay`          | 0.01           | AdamW weight decay.                                         |
| `--grad-clip`             | 1.0            | gradient-norm clip.                                        |
| `--warmup-steps`          | 100            | linear-warmup steps.                                       |
| `--token2json-format`     | true           | encode every field as `<s_x>v or <missing></s_x>` (parseable) vs legacy. |
| `--output-dir`            | checkpoints    | parent of the `best/` and `last/` checkpoint dirs.         |
| `--mlflow-experiment`     | None           | set a name to log params/metrics to MLflow.                |
| `--smoke`             | false          | tiny offline model, CPU, 64×64, few samples — for CI.       |

Weights load in fp32 (stable master weights + optimizer state); with
`--precision bf16` the forward runs under bf16 autocast, so the `donut` accel
kernels still execute in bf16 — speed without bf16-master-weight instability.

Training follows the canonical Donut convention: the task token `<s_donut>` is
the decoder start, so labels are `fields + eos` and `predict.py` seeds generation
with the task token.

Checkpoints are saved as HuggingFace `save_pretrained` directories — `best/`
(lowest val loss) and `last/` — each holding the model, the processor (with the
registered field tokens and image size), and a small `train_meta.json`. So
`predict.py` rebuilds the exact model from the dir without
re-passing flags.

---

## How quality is measured

For every `(document, field)` pair the model is scored against ground truth:

| outcome | meaning                                              |
|---------|------------------------------------------------------|
| **TP**  | GT has a value, model predicted it, value correct.   |
| **FP**  | GT has no value, model predicted something (hallucination). |
| **FN**  | GT has a value, model was wrong or silent.           |
| **TN**  | GT has no value, model correctly stayed silent.      |

From these: **precision** = TP/(TP+FP), **recall** = TP/(TP+FN), **F1** = their
harmonic mean. Reported in two modes:

- **strict** — exact string match.
- **soft** — normalized match (lowercase + trimmed whitespace).

Documents are also bucketed: `perfect` (no FP, no FN), `fn_only`, `fp_only`,
`mixed`. `predict.py` prints the full per-field tables for both modes.

---

## Do the donut accelerations speed up *training*?

The `donut` package's accel backends speed up inference. Whether they speed up
**training** is a separate question, answered with two measurements.

**Metrics — `docs/s = batch_size / Δt`**, where one *doc* = one image + its label
sequence. The only variable is the time window Δt:

- **e2e docs/s** — full step wall-time (data fetch + H2D + fwd + bwd + opt). Practical
  training throughput.
- **compute docs/s** — fwd + bwd + opt only, GPU-synced, no data loading.
  Hardware-limited throughput.
- **encoder docs/s** — encoder forward only; isolates the Swin SDPA patch.
- **data-bound %** = data-fetch / step. If high, the dataloader is the bottleneck and
  the backend can't move the wall clock.

**1. `bench_train.py` — is the compute faster, and where?** Strips the dataloader (one
fixed in-memory batch, reused) and times a training step per backend, broken into
encoder / decoder fwd, backward, optimizer step, with peak memory. Prints the actual
attn impl per component (fact, not assumption).

```bash
uv run python bench_train.py --backends baseline,eager,sdpa,fa \
  --image-height 1280 --image-width 960 --batch-size 4
# add --data-json /path  to also probe real dataloader throughput
# add --tiny             to run the harness on CPU with no downloads
```

**2. `train.py` per-epoch line — does it matter end-to-end?** Every epoch prints
`e2e`, `compute`, `data-bound %`, and `val` docs/s, and `train.py` asserts the backend
is live (`check_accel`) + prints the attn impls. If `bench_train.py` shows compute got
faster but the training `data-bound %` is high, the optimizations help the GPU but not
the wall clock — the real lever is the dataloader, not the kernel.
