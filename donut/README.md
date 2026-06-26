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

A separate question from inference. Two measurements, see
[`METRICS.md`](METRICS.md) for exact definitions:

```bash
# Mechanism: per-backend training-step breakdown, dataloader stripped.
uv run python scripts/train/bench_train.py --backends baseline,eager,sdpa,fa
#   --image-sizes 1280x960,1920x1440 --batch-sizes 1,4   sweep size/batch (like bench_speed)
#   add --probe-data-json /path  to also probe real dataloader throughput
#   add --tiny             to run the harness on CPU with no downloads

# End-to-end: train.py prints e2e / compute / data-bound % docs/s per epoch.
```

The atomic timer is `donut.bench.bench_train_step` (twin of the inference
`bench_infer_step`, same docs/s metric);
if compute got faster but `data-bound %` is high, the dataloader — not the
kernel — is the wall-clock lever.
