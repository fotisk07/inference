# `donut_train` — fine-tuning & quality ablation

Fine-tune Donut for **field extraction** and measure how good it is. Where the
`donut` package answers "how *fast* is the model," this one answers "how
*accurate* is it" — and how that accuracy depends on image size and batch size.
The two are joined into a speed-vs-accuracy decision in `notebooks/pareto.ipynb`.

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
# 1. Smoke test — tiny offline model on CPU, seconds, no downloads. Proves the
#    pipeline works end-to-end. Zero exit = OK.
uv run python train.py --smoke true

# 2. Real fine-tune (downloads donut-base the first time).
uv run python train.py --data_json /path/train.json --device cuda \
  --image_size '[1280,960]' --batch_size 4 --max_epochs 30 --seed 42

# 3. Score a saved checkpoint on labelled data (prints P/R/F1 tables).
uv run python predict.py --checkpoint checkpoints/epoch_030_val0.09.pt \
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
| `--backend`           | sdpa           | inference/attention backend from `donut` (eager/sdpa/fa).   |
| `--seed`              | None           | set it (e.g. 42) for reproducible shuffles/splits.          |
| `--weight_decay`      | 0.01           | AdamW weight decay.                                         |
| `--grad_clip`         | 1.0            | gradient-norm clip.                                        |
| `--warmup_steps`      | 100            | linear-warmup steps.                                       |
| `--token2json_format` | true           | encode every field as `<s_x>v or <missing></s_x>` (parseable) vs legacy. |
| `--output_dir`        | checkpoints    | where `epoch_NNN_valX.pt` checkpoints go.                  |
| `--ablation_out`      | None           | **write a one-run F1 summary JSON here** (see below).       |
| `--mlflow_experiment` | None           | set a name to log params/metrics to MLflow.                |
| `--smoke`             | false          | tiny offline model, CPU, 64×64, few samples — for CI.       |

Each checkpoint embeds the metadata (`model_name`, `image_size`,
`token2json_format`, …) so `predict.py` can rebuild the exact model without
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
`mixed`. `predict.py` prints the full per-field tables; the ablation summary
keeps the micro-averaged F1 (counts summed across all fields first).

---

## Ablation workflow — image size & batch size

Add `--ablation_out path.json` to any run and, after training, it scores the
validation split and writes a self-describing summary:

```jsonc
{
  "image_size": [1280, 960], "batch_size": 4, "lr": 0.0003, "backend": "sdpa",
  "n_train": 800, "final_val_loss": 0.09, "docs_per_sec": 12.4,
  "f1_strict": 0.91, "f1_soft": 0.94,
  "quality": { "strict": {...}, "soft": {...}, "doc_stats": {...} },
  "config": { ...full config... }
}
```

The launcher scripts run a sweep, one summary per setting, into
`results/ablation/`:

```bash
# Image size is the central question: how much resolution does training need?
DATA_JSON=/path/train.json DEVICE=cuda \
  IMAGE_SIZES=640x480,960x720,1280x960 \
  bash scripts/exp_image_size.sh        # → results/ablation/imgsize_<HxW>.json

# Which batch size trains best (and fastest) on this data?
DATA_JSON=/path/train.json DEVICE=cuda \
  BATCH_SIZES=1,2,4,8 \
  bash scripts/exp_batch_size.sh        # → results/ablation/batch_<bs>.json
```

Both accept the same env vars as overrides (`LR`, `MAX_EPOCHS`, `BACKEND`,
`SEED`, `OUT_DIR`, …). They default to the `test_data` smoke set; set
`SMOKE=true` to dry-run the whole sweep on the tiny offline model first.

> Use the **same `HxW` values** here as in `donut/scripts/run_ablation.sh` — the
> Pareto join keys on image size.

---

## Interpret

- **`notebooks/ablation.ipynb`** — loads `results/ablation/*.json` and plots
  **F1 vs image size** (find the smallest resolution that still reads the
  document) and **F1 / throughput vs batch size**, plus a ranked table.
- **`notebooks/pareto.ipynb`** — joins these F1 numbers with the inference
  latency from `donut/scripts/results/ablation/bench_speed.json` on image size,
  draws the **F1-vs-latency Pareto frontier**, and gives a decision table:
  *the smallest/fastest image that clears your F1 floor*. This is how you pick
  the final model.

Both notebooks are generated by `notebooks/_build_notebooks.py` (kept in-repo so
they're diffable); regenerate with `uv run python notebooks/_build_notebooks.py`.

---

## End-to-end recipe for "the best model"

```bash
# 1. Speed: backends × image size (in the donut package)
cd ../donut/scripts && IMAGE_SIZES=640x480,960x720,1280x960 bash run_ablation.sh

# 2. Accuracy: the SAME image sizes (here)
cd ../../donut_train
DATA_JSON=/path/train.json DEVICE=cuda \
  IMAGE_SIZES=640x480,960x720,1280x960 bash scripts/exp_image_size.sh

# 3. Decide: open notebooks/pareto.ipynb → pick the knee of the frontier.
```
