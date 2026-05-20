"""
Benchmark: float16 vs float32 operations on CPU, and the cost of uncached mask creation.

This reproduces the exact conditions that caused the Swin Transformer encoder to run
~55x slower on the H100 than on the A100. Run on both machines to compare.

Covers all four Swin stages so you can see the mask-creation cost at each spatial
resolution, not just the largest (stage 0).

Standalone -- no local imports. Requires: torch
"""

# /// script
# dependencies = [
#   "torch"
# ]
# ///
import sys
import time

import torch

WINDOW_SIZE = 8
SHIFT_SIZE = 4
ITERS = 10

# Swin stage dimensions for a 1280x960 input (patch_size=4, window_size=8).
# Each stage downsamples by 2x via patch merging. Widths that are not a
# multiple of window_size are padded up by maybe_pad() before mask creation,
# so the mask dimensions use the padded values (stages 2 and 3).
STAGES = [
    ("stage 0", 320, 240),  # 1280/4 x 960/4; 240 = 30*8 -- no padding
    ("stage 1", 160, 120),  # 120 = 15*8 -- no padding
    ("stage 2",  80,  64),  # raw W=60, padded to 64 (next multiple of 8)
    ("stage 3",  40,  32),  # raw W=30, padded to 32 (next multiple of 8)
]

H0, W0 = STAGES[0][1], STAGES[0][2]


def mask_shape(h, w):
    nw = (h // WINDOW_SIZE) * (w // WINDOW_SIZE)
    ww = WINDOW_SIZE * WINDOW_SIZE
    return (nw, ww, ww)


MASK_SHAPE_S0 = mask_shape(H0, W0)


def bench(label: str, fn, iters: int = ITERS, width: int = 52) -> float:
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {label:<{width}} {ms:8.2f} ms")
    return ms


def make_mask(h: int, w: int, dtype: torch.dtype) -> torch.Tensor:
    """Reproduces DonutSwinLayer.get_attn_mask exactly, with configurable h/w/dtype."""
    img_mask = torch.zeros((1, h, w, 1), dtype=dtype)
    cnt = 0
    ws, ss = WINDOW_SIZE, SHIFT_SIZE
    for sh in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
        for sw in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
            img_mask[:, sh, sw, :] = cnt
            cnt += 1
    mw = img_mask.view(1, h // ws, ws, w // ws, ws, 1)
    mw = mw.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws)
    mask = mw.unsqueeze(1) - mw.unsqueeze(2)
    mask = mask.masked_fill(mask != 0, float(-100.0))
    mask = mask.masked_fill(mask == 0, float(0.0))
    return mask


def separator(title: str = "") -> None:
    if title:
        print(f"\n{'=' * 20} {title} {'=' * 20}")
    else:
        print()


# == Header ==================================================================

print("=" * 72)
print("  CPU float16 vs float32 benchmark -- Swin attention mask generation")
print("=" * 72)
print(f"\n  Python   : {sys.version.split()[0]}")
print(f"  PyTorch  : {torch.__version__}")
if torch.cuda.is_available():
    print(f"  GPU      : {torch.cuda.get_device_name(0)}")
    print(f"  CUDA     : {torch.version.cuda}")
kb = MASK_SHAPE_S0[0] * MASK_SHAPE_S0[1] * MASK_SHAPE_S0[2] * 2 / 1024
print(f"\n  Stage-0 mask shape : {MASK_SHAPE_S0}  ({kb:.0f} KB as float16)")
print(f"  Iterations : {ITERS} per benchmark\n")

# == Part 1: masked_fill in isolation ========================================

separator("Part 1: masked_fill in isolation (the hot path in get_attn_mask)")

t_f32 = torch.zeros(MASK_SHAPE_S0)
t_f16 = torch.zeros(MASK_SHAPE_S0, dtype=torch.float16)

print(f"  Tensor shape: {list(MASK_SHAPE_S0)}  (stage 0 -- largest mask)")
ms_f32 = bench(
    "masked_fill  float32 on CPU", lambda: t_f32.masked_fill(t_f32 != 0, -100.0)
)
ms_f16 = bench(
    "masked_fill  float16 on CPU", lambda: t_f16.masked_fill(t_f16 != 0, -100.0)
)
print(f"\n  -> float16 is {ms_f16 / ms_f32:.0f}x slower than float32 on this CPU")

if torch.cuda.is_available():
    t_f16_gpu = t_f16.cuda()
    ms_gpu = bench(
        "masked_fill  float16 on GPU",
        lambda: t_f16_gpu.masked_fill(t_f16_gpu != 0, -100.0),
    )
    torch.cuda.synchronize()
    print(f"  -> GPU float16 is {ms_f16 / ms_gpu:.0f}x faster than CPU float16")

# == Part 2: full get_attn_mask -- all four stages ===========================

separator("Part 2: full get_attn_mask -- all four Swin stages")

print("  This is what transformers 4.37.2 runs on every forward pass.\n")
print(
    f"  {'stage':<10} {'mask shape':<22} {'float32':>10}   {'float16':>10}   {'slowdown':>8}"
)
print("  " + "-" * 64)

stage_ms_f16 = {}
stage_ms_f32 = {}

for stage_name, H, W in STAGES:
    sh = mask_shape(H, W)
    ms32 = None
    ms16 = None

    # warmup
    for _ in range(3):
        make_mask(H, W, torch.float32)
        make_mask(H, W, torch.float16)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        make_mask(H, W, torch.float32)
    ms32 = (time.perf_counter() - t0) / ITERS * 1000

    t0 = time.perf_counter()
    for _ in range(ITERS):
        make_mask(H, W, torch.float16)
    ms16 = (time.perf_counter() - t0) / ITERS * 1000

    slowdown = ms16 / ms32 if ms32 > 0 else 0
    print(
        f"  {stage_name:<10} {str(sh):<22} {ms32:10.2f}ms  {ms16:10.2f}ms  {slowdown:>6.0f}x"
    )

    stage_ms_f16[stage_name] = ms16
    stage_ms_f32[stage_name] = ms32

# How many shifted blocks per stage (every other block is shifted)
blocks_per_stage = {"stage 0": 1, "stage 1": 1, "stage 2": 7, "stage 3": 1}
total_f16 = sum(blocks_per_stage[n] * stage_ms_f16[n] for n in blocks_per_stage)
total_f32 = sum(blocks_per_stage[n] * stage_ms_f32[n] for n in blocks_per_stage)

print(
    f"\n  Total mask cost per forward pass ({sum(blocks_per_stage.values())} shifted blocks):"
)
print(f"    original (float16, recomputed) : {total_f16:7.1f} ms")
print(
    f"    patched  (float32, first call) : {total_f32:7.1f} ms  (then cached -> ~0 ms)"
)

# == Part 3: caching impact ==================================================

separator("Part 3: caching -- cost over N forward passes (stage 0 mask, largest)")


def simulate_original(n_passes: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n_passes):
        make_mask(H0, W0, torch.float16)
    return (time.perf_counter() - t0) * 1000


def simulate_patched(n_passes: int, device="cpu") -> float:
    cache = {}
    key = (H0, W0, torch.float16)
    t0 = time.perf_counter()
    for _ in range(n_passes):
        if key not in cache:
            m = make_mask(H0, W0, torch.float32)
            if device == "cuda" and torch.cuda.is_available():
                cache[key] = m.to(device=device, dtype=torch.float16)
            else:
                cache[key] = m.to(dtype=torch.float16)
        _ = cache[key]
    return (time.perf_counter() - t0) * 1000


print(
    "  Swin stage 2 has 14 blocks, 7 with shifted windows -> 7 mask calls per forward pass."
)
print("  Multiply by batch size and number of images for total cost.\n")

for n in (1, 7, 50, 200):
    t_orig = simulate_original(n)
    t_patch = simulate_patched(n, device="cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"  {n:>4} forward passes -- original: {t_orig:7.1f} ms   patched: {t_patch:6.2f} ms   speedup: {t_orig / max(t_patch, 0.001):.0f}x"
    )

# == Part 4: dtype conversion CPU vs GPU =====================================

ms_mask_f16 = stage_ms_f16["stage 0"]
ms_mask_f32 = stage_ms_f32["stage 0"]

if torch.cuda.is_available():
    separator("Part 4: float32 -> float16 conversion -- CPU vs GPU")

    mask_f32_cpu = make_mask(H0, W0, torch.float32)
    mask_f32_gpu = mask_f32_cpu.cuda()

    print(f"  Source tensor: float32, shape {list(mask_f32_cpu.shape)}")
    ms_cpu_cast = bench(
        "float32 -> float16  on CPU", lambda: mask_f32_cpu.to(dtype=torch.float16)
    )
    ms_gpu_cast = bench(
        "float32 -> float16  on GPU", lambda: mask_f32_gpu.to(dtype=torch.float16)
    )
    torch.cuda.synchronize()
    print(
        f"\n  -> Doing the dtype conversion on GPU is {ms_cpu_cast / ms_gpu_cast:.0f}x faster"
    )
    print(
        "    The patched code sends float32 to GPU and converts there (.to(device, dtype=...))"
    )

# == Summary =================================================================

separator("Summary")
print(f"  Stage-0 get_attn_mask float16 (original) : {ms_mask_f16:7.1f} ms per call")
print(
    f"  Stage-0 get_attn_mask float32 (patched)  : {ms_mask_f32:7.1f} ms  (first call only, then cached)"
)
print("  Subsequent calls (cached GPU)            :    ~0.00 ms")
print()
print(
    f"  Total mask overhead per forward pass     : {total_f16:7.1f} ms original -> {total_f32:5.1f} ms patched (first pass)"
)
print()
print("  Root cause: CPUs without AVX-512 FP16 extensions emulate float16 in software.")
print("  masked_fill on a (1200, 64, 64) float16 tensor touches 4.9M elements with no")
print("  SIMD acceleration -- hence ~500ms on an older server CPU.")
print()
print("  Fix: compute in float32, convert once on GPU, cache the result per block.")
