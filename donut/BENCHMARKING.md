# Benchmarking & ablation — `donut`

How to measure the **inference speed** of the Donut model and decide which
acceleration backend / batch size / image size is fastest. This package is
speed-only: it times the model on synthetic tensors. Accuracy lives in
`donut_train`; the two are joined in `donut_train/notebooks/pareto.ipynb`.

---

## Mental model

A single set of Donut weights, run under different **backends** (same outputs,
different attention implementation):

| backend    | what it is                                                        |
|------------|-------------------------------------------------------------------|
| `baseline` | nothing applied — not even mask caching. The speedup reference.    |
| `eager`    | mask caching only (shifted-window masks cached per feature shape). |
| `sdpa`     | mask cache + PyTorch `scaled_dot_product_attention` (enc + dec).   |
| `fa`       | mask cache + SDPA encoder + flash-attention decoder (CUDA only).   |

The benchmark sweeps **backend × image_size × batch_size** and, for each cell,
times the encoder forward pass and a full `generate()` call.

**Why image size is a dimension.** Donut's Swin encoder is resolution-agnostic
(convolutional patches + per-window relative bias), so the same weights run at
any `H×W` divisible by 40. Bigger images = more patches = more compute. This
benchmark shows the *cost*; it cannot show the *accuracy* trade-off (that needs
real images — see `donut_train`).

---

## Run it

```bash
cd donut/scripts
bash run_ablation.sh                       # full default grid → results/ablation/
```

Every knob is an environment variable:

```bash
IMAGE_SIZES=640x480,1280x960 \
BACKENDS=eager,sdpa,fa \
BATCH_SIZES=1,2,4,8 \
MAX_NEW_TOKENS=128 \
GEN_MODE=fixed \
N_RUNS=30 N_WARMUP=5 \
OUT=results/ablation \
bash run_ablation.sh
```

Then open `donut/notebooks/ablation.ipynb` to interpret.

### Or call the script directly

```bash
uv run python scripts/bench_speed.py \
  --image-sizes 640x480,960x720,1280x960 \
  --backends eager,sdpa,fa \
  --batch-sizes 1,2,4 \
  --max-new-tokens 128 \
  --gen-mode fixed \
  --n-runs 30 --n-warmup 5 \
  --out results/ablation
```

| flag                | meaning                                                              |
|---------------------|----------------------------------------------------------------------|
| `--image-sizes`     | comma-separated `HxW`. Each H,W must be divisible by 40.              |
| `--backends`        | which presets to time (`baseline` is always added automatically).    |
| `--batch-sizes`     | comma-separated batch sizes.                                         |
| `--max-new-tokens`  | tokens to decode per image in `generate()`.                          |
| `--gen-mode`        | `fixed` or `eos` — see below.                                       |
| `--n-runs`          | timed iterations per cell (more = tighter confidence interval).      |
| `--n-warmup`        | discarded warmup iterations before timing.                          |
| `--tiny`            | tiny random model, CPU, no downloads — offline smoke test.          |
| `--device`          | `cuda` / `cpu` (default: auto).                                      |
| `--dtype`           | `bf16` / `f16` / `f32` (default: bf16 on cuda, f32 on cpu).          |
| `--out`             | output directory for `bench_speed.json`.                            |

**Smoke check** (seconds, no GPU, no downloads):

```bash
uv run python scripts/bench_speed.py --tiny --backends eager,sdpa \
  --image-sizes 64x64 --n-runs 3 --out results/smoke
```

### `--gen-mode`: fixed vs eos

- **`fixed`** (default): always emit exactly `--max-new-tokens`
  (`min_new_tokens == max_new_tokens`). Clean, content-independent per-step
  timing. Best for comparing backends apples-to-apples.
- **`eos`**: stop naturally at the EOS token (capped by `--max-new-tokens`), so
  latency reflects a content-dependent decode length. On *synthetic* pixels the
  stopping point is model noise, so `eos` is only meaningful when you set
  `--max-new-tokens` to a representative real output length. The realized mean
  token count is recorded as `generate.new_tokens` and used for throughput.

---

## Output — `bench_speed.json`

```jsonc
{
  "meta": { "model_id", "device", "dtype", "torch", "transformers",
            "seed", "timestamp", "git_sha" },
  "records": [
    {
      "image_height": 1280, "image_width": 960,
      "backend": "sdpa", "batch_size": 1, "gen_mode": "fixed",
      "encoder":  { "mean_ms", "std_ms", "p50_ms", "p95_ms", "n_runs" },
      "generate": { "mean_ms", "std_ms", "p50_ms", "p95_ms", "n_runs",
                    "new_tokens" },
      "throughput": { "images_per_s", "tokens_per_s" },
      "speedup_vs_baseline": { "encoder", "generate" }   // vs baseline, same size+batch
    }
    // ... one record per (image_size × backend × batch_size)
  ]
}
```

- **`speedup_vs_baseline`** is computed *within each image-size group*, so you
  can ask "does acceleration help more at higher resolution?"
- **`p95_ms`** is the tail latency; **`std_ms`** drives the confidence intervals
  in the notebook (`1.96·σ/√n`).

`pd.json_normalize(raw["records"])` flattens it for pandas.

---

## Interpret — `donut/notebooks/ablation.ipynb`

Question-driven cells: fastest backend per resolution, whether speedup grows
with image size, the throughput-maximizing batch size, and where time goes
(encoder vs decoder), ending in a ranked decision table. `bench.ipynb` is a
simpler view of the same JSON.

---

## Correctness (separate from speed)

Speed numbers are only trustworthy if the backend produces the right outputs.
Every backend is structurally verified with `check_accel` *before* timing
(`bench_speed.py` refuses to bench an unverified config), and the
`scripts/audit_*.py` tools quantify numerical divergence vs eager. `pytest
tests/` proves apply/revert round-trips and output equivalence on the tiny
model in ~1s.

> **Caveat — synthetic data.** This package feeds random pixels, so it measures
> compute cost, never accuracy. "Smaller image is faster" is always true here;
> whether a smaller image is *good enough* is answered in `donut_train`.
