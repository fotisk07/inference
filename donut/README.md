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
