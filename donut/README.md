# donut

Donut (`naver-clova-ix/donut-base-finetuned-cord-v2`, Swin encoder + MBart
decoder) with toggleable, **auditable** inference accelerations.

Design goal: every optimization is a plain module with three functions —
apply, revert, check — so each one can be independently verified against the
eager baseline, on the same model instance, with synthetic data and no
downloads.

## Usage

```python
from donut import load_model, apply_accel, revert_accel, check_accel

model, processor = load_model()                  # auto device/dtype, sdpa backend
model, processor = load_model(backend="eager")   # mask caching only
model, processor = load_model(backend="fa")      # flash-attn decoder (CUDA)

apply_accel(model, "sdpa")    # apply a preset in-place (idempotent)
check_accel(model, "sdpa")    # assert it is structurally active
revert_accel(model)           # restore the exact eager baseline
```

Backends (`donut.accel.PRESETS`):

| backend | steps |
|---------|-------|
| `eager` | mask cache |
| `sdpa`  | mask cache + encoder SDPA patch + decoder SDPA dispatch |
| `fa`    | mask cache + encoder SDPA patch + decoder flash-attention dispatch |

`fa` picks `flash_attention_2` or `flash_attention_4` from what is installed
and raises if neither is available (no silent fallback).

## Layout

```
src/donut/
  accel/            one module per optimization: apply_x / revert_x / check_x
  synthetic.py      synthetic inputs + tiny offline model (tests, smoke runs)
  audit.py          diff stats, per-layer capture, stepwise decode comparison
  bench.py          latency timing helpers
tests/              pytest suite — CPU, offline, ~1s (tiny random model)
scripts/            audit + benchmark CLIs; save to results/, print a summary
notebooks/          audit.ipynb / bench.ipynb — visualize results/
```

For running the speed sweep (backends × image size × batch size) and reading the
results, see **[BENCHMARKING.md](BENCHMARKING.md)**.

## Auditing workflow

Every optimization is verifiable at three levels:

1. **Structural** — `check_accel(model, backend)`: the patch is actually in
   place (per-block guard attributes, decoder config flags).
2. **Numerical** — `uv run pytest`: KV-cache correctness (cached vs full
   re-forward logits), mask cache bit-exactness, SDPA-vs-eager closeness,
   token-sequence equality. Runs on CPU in ~1s against a tiny random
   DonutSwin+MBart, no downloads.
3. **Empirical** — the scripts, on the real checkpoint with synthetic inputs:

```bash
uv run python scripts/audit_encoder.py            # encoder diff: eager vs SDPA
uv run python scripts/audit_layers.py             # where divergence enters, per block
uv run python scripts/audit_decoder.py --encoder eager   # decoder-intrinsic diff
uv run python scripts/audit_decoder.py --encoder sdpa    # end-to-end accumulation
uv run python scripts/bench_speed.py --backends eager,sdpa,fa --batch-sizes 1,2,4
```

Each script saves to `results/` (JSON/CSV/npz, with model/dtype/versions/seed
metadata) and prints a summary. Add `--tiny` to any script for an offline
smoke run. Interpret the outputs with `notebooks/audit.ipynb` and
`notebooks/bench.ipynb`.

### The known SDPA encoder divergence

Donut's Swin uses 10×10 windows → attention seq_len 100, not divisible by 8,
so the Flash SDP kernel is unavailable and PyTorch falls back to
Efficient/Math SDP, which accumulates in float32 while eager computes in
bfloat16. The SDPA path is *more* precise but different: small encoder diffs
that propagate into decoder logits. `audit_encoder.py` quantifies the diff,
`audit_layers.py` localizes it, and `audit_decoder.py` measures whether it
ever flips a decoded token (the true correctness gate).

## Extending: adding a new acceleration

An optimization is one module in `src/donut/accel/` with three functions:

```python
# src/donut/accel/my_opt.py
def apply_my_opt(model) -> None:
    """Idempotent: guard with an attribute so double-apply is a no-op."""
    if getattr(model, "_my_opt_applied", False):
        return
    ...  # patch / wrap / flip config; save what you need to undo it
    model._my_opt_applied = True

def revert_my_opt(model) -> None:
    """No-op when not applied; must restore the exact pre-apply state."""
    if not getattr(model, "_my_opt_applied", False):
        return
    ...
    del model._my_opt_applied

def check_my_opt(model) -> None:
    """Raise AssertionError (with location detail) when not active."""
    assert getattr(model, "_my_opt_applied", False), "my_opt is not applied"
```

Then wire and verify it:

1. Add a step in `src/donut/accel/__init__.py`:
   `MY_OPT = (apply_my_opt, revert_my_opt, check_my_opt)` and put it in an
   existing or new `PRESETS` entry. Order matters: mask cache first (the SDPA
   encoder patch consumes its cached bias).
2. Add a `tests/test_numerical.py` case proving output equivalence (or
   bounded divergence) on the tiny fixture, and rely on
   `test_accel_apply_revert.py` to cover the apply/check/revert round-trip by
   adding the new backend to its `BACKENDS` list.
3. Run the audit scripts before/after on the real model; `bench_speed.py
   --backends eager,sdpa,<new>` will refuse to time it unless `check_accel`
   passes.

Because apply/revert restore exact state, every new technique is measured
against the same in-memory baseline — no model reloads, no cross-contamination
between configs.

### Future directions

- **Other encoder attention backends** — FlexAttention or xformers via the
  same `DonutSwinSelfAttention.forward` patch point as `encoder_sdpa.py`; the
  audit scripts directly measure whether they pay off numerically.
- **Window padding to unlock Flash SDP** — pad windows 10×10 → 100→128 tokens
  so seq_len % 8 == 0; needs relative-position-bias re-interpolation, and
  `audit_layers.py` would quantify the numerical cost.
- **Quantization** — int8/fp8 weights via torchao as a load-time step
  (revert = reload); decoder lm_head and FFNs are the obvious targets.
- **Static KV cache + CUDA graphs** — `generate(cache_implementation="static")`
  to eliminate per-step launch overhead at TPOT.
- **Batching** — continuous batching of decoder steps; `bench_speed.py
  --batch-sizes` already measures the scaling headroom.
