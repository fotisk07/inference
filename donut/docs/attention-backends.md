# Attention backends in this repo: presets, regimes, and why FA isn't winning

This explains the terms used when discussing `scripts/bench_speed.py` /
`scripts/bench_attention_kernels.py` results: presets, decode vs prefill,
SDPA backends, and why flash attention isn't beating SDPA on H100 for this
model.

## The basic operation: attention

Every transformer layer does the same core math: given a query vector and a
set of key/value vectors, compute how much the query should "attend to"
each key, and produce a weighted sum of the values. This is
`scaled_dot_product_attention` (SDPA) — `softmax(Q @ Kᵀ / sqrt(d)) @ V`.

There isn't one way to compute this on a GPU. Different *kernels*
(implementations of the same math) trade memory, parallelism, and
specialization differently. That's what this whole investigation is about:
which kernel is fastest, under which conditions.

## What a "preset" is in this codebase

`src/donut/accel/` defines small, independent optimizations. Each one is a
module with three functions:

- `apply_x(model)` — patch the model in place to use this optimization
- `revert_x(model)` — undo the patch
- `check_x(model)` — assert the optimization is actually active

A **preset** (`PRESETS` dict in `src/donut/accel/__init__.py`) is just a
named list of these steps applied together. `--backends` in
`scripts/bench_speed.py` picks which presets to benchmark. There are five:

| preset | encoder kernel | decoder kernel |
|---|---|---|
| `baseline` | eager (plain PyTorch ops, nothing patched) | eager |
| `eager` | eager + cached attention-mask bias (`mask_cache.py`) | eager |
| `sdpa` | SDPA, auto-picked backend | SDPA, auto-picked backend |
| `sdpa_cudnn` | SDPA, auto-picked backend | SDPA, **forced to the cuDNN backend** |
| `fa` | SDPA, auto-picked backend | the actual FlashAttention-4 kernel |

The encoder (DonutSwin) has no flash-attention path at all — it's a legacy
class with its own hand-written attention, so `"sdpa"` and `"fa"` both patch
it the same way (`encoder_sdpa.py`). The only thing that changes between
`sdpa` / `sdpa_cudnn` / `fa` is **which kernel the decoder uses**. That's
intentional: it isolates the decoder-kernel comparison from everything else.

## SDPA backends: math / efficient / flash / cudnn

`F.scaled_dot_product_attention` (what `"sdpa"` and `"sdpa_cudnn"` both call
under the hood) isn't itself a kernel — it's a dispatcher. PyTorch picks one
of several backends based on input shape/dtype/mask:

- **math** — naive, literal `Q @ Kᵀ`, softmax, `@ V`. Always works, never
  fastest. The reference/fallback.
- **efficient** — memory-efficient attention (xformers-style), broader
  compatibility (e.g. handles arbitrary float attention masks, which the
  encoder needs for its window bias).
- **flash** — PyTorch's *own* built-in flash-attention kernel. Note: **this
  is a different implementation from the `flash-attn-4` package** used by
  the `"fa"` preset — same algorithm family, different code, can perform
  differently.
- **cudnn** — NVIDIA's cuDNN attention kernel. Only available on supported
  GPUs (H100 qualifies).

Normally PyTorch auto-picks one of these per call based on the shapes
involved, and you can't easily tell which one it picked. `donut.accel.sdpa_backend(name)`
(`src/donut/accel/sdpa_backend.py`) is a context manager that *forces* one
specific backend for whatever call it wraps — `"math"`, `"efficient"`,
`"flash"`, or `"cudnn"`. The `sdpa_cudnn` preset uses it to pin the decoder
to cuDNN specifically, instead of trusting the auto heuristic.

## decode vs prefill: why they're completely different workloads

This is the part that explains the confusing benchmark results.

- **prefill**: process a sequence of N tokens all at once (e.g. the first
  forward pass over a prompt). `query_len == kv_len == N`. This is what
  training and the "first pass" of generation typically look like — lots of
  parallel work per kernel call. Flash attention's tiled-block algorithm is
  *designed* for this: the bigger N is, the more it wins by avoiding
  materializing the full N×N attention matrix.
- **decode**: generate one new token at a time, reusing a KV cache of
  everything generated so far. `query_len == 1`, `kv_len` grows by one each
  step. There's no parallelism across query positions to exploit — each
  kernel call does one query attending to a (possibly long) cache — so a
  kernel's fixed launch/setup overhead dominates instead of its asymptotic
  algorithmic advantage.

**Donut's `generate()` call in this repo is decode-only.**
`make_decoder_input_ids` (`src/donut/synthetic.py`) builds a 1-token BOS
prompt, so with `use_cache=True`, every single step of `bench_generate`
(`src/donut/bench.py`) is a `query_len=1` attention call. There is no
prefill phase here at all — flash attention never gets to run in the regime
it's good at.

## What the H100 numbers actually showed

`scripts/bench_attention_kernels.py` benchmarks the real FA4 kernel
(`flash_attn.cute.flash_attn_func`, the exact function the `"fa"` preset
dispatches to) against SDPA's backends, sweeping `kv_len`, `batch_size`, and
`mode` (`decode` = query_len 1, `prefill` = query_len == kv_len, causal) —
fully decoupled from the Donut model, just raw tensors at the decoder's real
shape (16 heads, head_dim 64, bf16). A real run on H100 showed:

- **decode mode**: FA4 never wins a single row. At `kv_len=1, bs=1`:
  FA4 = 0.086 ms vs SDPA-flash = 0.04 ms, SDPA-efficient = 0.035 ms — FA4 is
  *2x slower*. Even at the largest decode shape tested (`kv_len=4096,
  bs=32`), `sdpa-cudnn` (0.202 ms) still beat FA4 (0.274 ms).
- **prefill mode**: FA4's advantage is real and large. At `kv_len=4096,
  bs=8`: FA4 = 0.77 ms vs `sdpa-math` = 47.1 ms (61x), vs `sdpa-efficient` =
  2.2 ms (3x).
- **`sdpa-cudnn` won or tied in almost every row**, including beating FA4 in
  FA4's own best case (`kv_len=4096, bs=32` prefill: cuDNN 2.43 ms vs
  FA4 2.90 ms).
- `sdpa-math` is the naive reference and behaves exactly as expected: fine
  at small shapes, catastrophic at large ones (OOM at `kv_len=4096, bs=32`
  prefill).

**Conclusion**: the `"fa"` preset isn't broken. FlashAttention's design
sweet spot (long-sequence, compute-bound, lots of query-side parallelism)
simply doesn't overlap with what Donut's `generate()` actually does
(decode-bound, `query_len=1` every step). Paying for FA4's kernel complexity
buys nothing in that regime, and can lose to simpler kernels on launch
overhead alone. `sdpa_cudnn` is the more promising decoder kernel for this
specific workload based on this data — that's why it's now a fifth preset
(`--backends ...,sdpa_cudnn,...` in `bench_speed.py`).

## Where to look for more

- `src/donut/accel/sdpa_backend.py` — the backend-forcing context manager.
- `src/donut/accel/decoder_sdpa.py` — `sdpa_cudnn` preset implementation
  (registers a custom `transformers` `AttentionInterface` entry that wraps
  the standard SDPA dispatch in the forced-backend context manager).
- `scripts/bench_attention_kernels.py` — the isolated kernel-level sweep
  that produced the numbers above; rerun with different `--kv-lens
  --batch-sizes --modes` to probe other shapes.
- `scripts/bench_speed.py` — full-model benchmark across all five presets;
  `notebooks/bench.ipynb` visualizes its output.
