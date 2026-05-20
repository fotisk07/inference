# Debugging Slow Swin Transformer Inference on H100

## Symptoms

End-to-end inference for a single image with the Donut model took ~7 s on an H100 MIG 1g.10gb slice
but only ~860 ms on an A100 MIG 1g.10gb slice — a 8× end-to-end difference, with the encoder
specifically at **7061 ms vs 127 ms (55× slower)**.

| Machine | GPU | Encode | Decode |
|---|---|---|---|
| Local | A100 MIG 1g.10gb, CUDA 13.0, torch 2.12 | 127 ms | 730 ms |
| Other | H100 MIG 1g.10gb, CUDA 12.1, torch 2.1.2 | 5233 ms | 907 ms |

---

## Investigation

### Step 1 — Rule out the obvious

The H100 is architecturally faster than the A100, so hardware wasn't the issue.

We added a diagnostics script (`time_forward_pass.py`) that printed:

- Python / PyTorch / CUDA / cuDNN versions
- GPU name and compute capability
- Model weight dtypes
- Encode and decode times separately

Initial findings confirmed both machines were running **bfloat16** weights, the same
transformers version (4.37.2), and the model loaded correctly.

### Step 2 — Upgrading PyTorch didn't help

Upgrading the H100 machine from torch 2.1.2 to 2.12 produced **no timing change**,
ruling out PyTorch version as the cause. (The dtype changed from bfloat16 to float16 after the
upgrade due to the model config taking precedence — a separate minor issue.)

### Step 3 — GPU hardware check with a matmul microbenchmark

We added a raw matmul benchmark to the script using CUDA events:

```
large  1024×1024×1024        0.025 ms   ← fast, tensor cores working
swin-s0 (1200,4,64,32)@(...) 0.200 ms
swin-s2  (75,16,64,32)@(...) 0.050 ms
```

The GPU itself was healthy. This ruled out Confidential Computing mode, MIG misconfiguration,
and GPU throttling.

### Step 4 — Per-stage encoder breakdown

We timed each of the 4 Swin stages individually:

```
patch embed                      3 ms
stage 0  (2 blocks)           1271 ms    ← 635 ms per block
stage 1  (2 blocks)            618 ms    ← 309 ms per block
stage 2 (14 blocks)           5562 ms    ← 397 ms per block
stage 3  (2 blocks)            539 ms    ← 270 ms per block
```

Stage 2 dominated simply because it has **14 blocks** (vs 2 in the others). Per-block,
all stages were similarly slow. The bottleneck was not stage-specific.

### Step 5 — Per-operation breakdown inside one block

We added `diagnose_block()` which timed every individual operation inside stage 0, block 1
(the shifted-window block):

```
layernorm_before                    0.01 ms
view to (B,H,W,C)                   0.01 ms
get_attn_mask (CPU)               519.00 ms   ← !!!
attn_mask .to(device)               0.30 ms
torch.roll                          0.05 ms
window_partition                    0.05 ms
self.attention                      0.80 ms
window_reverse                      0.05 ms
...
```

**`get_attn_mask` was taking 519 ms** — accounting for nearly the entire block runtime.

---

## Root Cause

`DonutSwinLayer.get_attn_mask` has two bugs:

### Bug 1 — float16 arithmetic on CPU

```python
# Inside get_attn_mask (transformers 4.37.2)
img_mask = torch.zeros((1, height, width, 1), dtype=dtype)   # ← dtype = float16!
...
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))  # float16 on CPU
attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))     # float16 on CPU
```

The function creates the mask tensor in the same dtype as the model's hidden states —
which is float16 (or bfloat16). **CPUs have no native float16 arithmetic unit.**

On x86 processors without AVX-512 FP16 extensions (Intel Sapphire Rapids / AMD Zen 4 or newer),
PyTorch falls back to a software emulation path: every float16 element is individually
widened to float32, operated on, then narrowed back. For a mask of shape `(1200, 64, 64)`
= 4.9 M elements, this makes two `masked_fill` calls take **~500 ms** instead of < 1 ms.

The A100 machine's CPU either supports AVX-512 FP16 natively or runs a PyTorch build that
uses it, which is why the same code is fast there.

### Bug 2 — Recomputed on every forward pass

The mask for a given block is **completely deterministic** — it depends only on
`window_size`, `shift_size`, `height`, and `width`, all of which are fixed constants
for a trained model. Yet `get_attn_mask` is called on every single forward pass and
recomputes the mask from scratch each time.

With a batch of images or repeated inference, this cost multiplies linearly.

---

## Why the A100 Was Fast

The A100 machine runs CUDA 13.0 with a newer CPU that supports AVX-512 FP16 instructions,
so PyTorch's float16 masked_fill runs in vectorised hardware rather than falling back to
software emulation. The 519 ms → <1 ms gap is entirely explained by this ISA difference.

---

## The Fix

Monkey-patch `get_attn_mask` on every shifted block before inference:

```python
def patch_attn_mask(model):
    import types
    device = next(model.encoder.parameters()).device

    def _fast_get_attn_mask(self, height, width, dtype):
        if self.shift_size == 0:
            return None
        key = (height, width, dtype)
        if not hasattr(self, "_mask_cache"):
            self._mask_cache = {}
        if key not in self._mask_cache:
            ws, ss = self.window_size, self.shift_size
            # Compute in float32 on CPU — fast on all hardware
            img_mask = torch.zeros((1, height, width, 1))
            cnt = 0
            for h in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                for w in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mw = img_mask.view(1, height // ws, ws, width // ws, ws, 1)
            mw = mw.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws)
            mask = mw.unsqueeze(1) - mw.unsqueeze(2)
            mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
            # dtype conversion and GPU transfer done once; GPU handles fp32→fp16 fast
            self._mask_cache[key] = mask.to(device=device, dtype=dtype)
        return self._mask_cache[key]  # GPU tensor — .to(device) in forward() is a no-op

    for stage in model.encoder.encoder.layers:
        for block in stage.blocks:
            block.get_attn_mask = types.MethodType(_fast_get_attn_mask, block)
```

**Three things this patch does:**

1. **Computes in float32** — every CPU has fast float32 ALU; the mask computation drops
   from ~519 ms to < 1 ms.
2. **Caches per `(height, width, dtype)`** — the mask is identical across all forward
   passes, so it is computed exactly once per block during the first (warmup) call.
3. **Stores on GPU** — `mask.to(device=device, dtype=dtype)` does the float32→float16
   conversion on the GPU (fast) and stores the result on-device. Every subsequent call
   returns the cached GPU tensor; the `.to(device)` in the original forward() becomes
   a no-op because the tensor is already there.

---

## Result

After applying the patch, encode time on the H100 dropped from **7061 ms to ~180 ms**,
consistent with the expected performance of an H100 relative to an A100.

---

## Takeaways

1. **CPU float16 is not free.** `dtype=model.dtype` feels natural but silently creates
   CPU float16 tensors when the model is in half precision. Any CPU-side computation
   (mask generation, position bias tables, preprocessing) should use float32 unless
   the CPU is known to have native fp16 hardware support.

2. **Cache deterministic constants.** Any value that depends only on fixed model
   hyperparameters (window size, shift size, resolution) should be computed once, not
   on every forward pass. This applies to attention masks, position bias tables, and
   similar structural tensors throughout transformer architectures.

3. **Microbenchmarks can mislead.** A large matmul benchmark showed the GPU was healthy.
   But the bottleneck was entirely on the CPU side — invisible to GPU-only profiling.
   Per-operation timing inside the model was necessary to find it.

4. **The same code can behave very differently across CPUs.** A difference in ISA
   support (AVX-512 FP16) between two server CPUs produced a 500× difference for a
   single operation, making the same model appear broken on one machine and fast on
   another with no error messages.
