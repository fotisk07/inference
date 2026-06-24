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

```bash
# 1. Smoke test — tiny offline model on CPU, seconds. Proves the
#    pipeline works end-to-end. Zero exit = OK.
uv run python train.py --smoke true

# 2. Real fine-tune (downloads donut-base the first time).
uv run python train.py --data_json /path/train.json --device cuda \
  --image_size '[1280,960]' --batch_size 4 --max_epochs 30 --seed 42

# 3. Score a saved checkpoint on labelled data (prints P/R/F1 tables).
#    --checkpoint is a saved dir: checkpoints/best (lowest val loss) or
#    checkpoints/last (final epoch). Add --output_json preds.json to dump
#    per-document {image, gt, pred} records.
uv run python predict.py --checkpoint checkpoints/best \
  --data_json /path/val.json
```

> **Note on `--image_size`.** It is a `(height, width)` tuple, so pass it as a
> JSON list: `--image_size '[1280,960]'`. `--image_size 1280 960` does **not**
> work.

---

## `train.py` — the knobs that matter

| flag                  | default        | meaning                                                        |
|-----------------------|----------------|----------------------------------------------------------------|
| `--data_json`         | test_data      | aggregate JSON described above.                               |
| `--val_split`         | 0.2            | fraction held out for validation.                            |
| `--image_size`        | `[1280,960]`   | `(H,W)` fed to the encoder. **Lower = faster + less VRAM, but less legible.** |
| `--batch_size`        | 4              | docs per step. Higher = faster/epoch but more VRAM.          |
| `--lr`                | 3e-4           | AdamW learning rate.                                         |
| `--max_epochs`        | 30             | training epochs.                                            |
| `--max_length`        | 128            | max decoder tokens (also the eval generation cap).          |
| `--backend`           | sdpa           | attention backend from `donut`, active in training (eager/sdpa/fa). |
| `--precision`         | bf16           | on CUDA: fp32 master weights + bf16 autocast compute. `fp32` for full fp32. |
| `--seed`              | None           | set it (e.g. 42) for reproducible shuffles/splits.          |
| `--weight_decay`      | 0.01           | AdamW weight decay.                                         |
| `--grad_clip`         | 1.0            | gradient-norm clip.                                        |
| `--warmup_steps`      | 100            | linear-warmup steps.                                       |
| `--token2json_format` | true           | encode every field as `<s_x>v or <missing></s_x>` (parseable) vs legacy. |
| `--output_dir`        | checkpoints    | parent of the `best/` and `last/` checkpoint dirs.         |
| `--mlflow_experiment` | None           | set a name to log params/metrics to MLflow.                |
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
