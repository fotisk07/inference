"""Model patches for Donut inference.

Two variants are provided:
  patch_attn_mask(model)      -- CPU float32 computation + H2D transfer. Safe for
                                  both CPU and GPU inference.
  patch_attn_mask_gpu(model)  -- all computation directly on GPU. ~75x faster
                                  cold-start; requires CUDA.

Both fix the same two bugs in DonutSwinLayer.get_attn_mask (transformers 4.37.2):
  1. Float16 arithmetic on CPU — falls back to software emulation on CPUs without
     AVX-512 FP16, making masked_fill ~500x slower than on GPU.
  2. Mask recomputed on every forward pass — it is a deterministic function of
     (window_size, shift_size, height, width) and should be computed once.

See H100_swin_debug_report.md for the full investigation and benchmark_fp16_cpu.py
Part 5 for the full cache x dtype/device ablation.
"""

import types

import torch


def patch_attn_mask(model) -> None:
    """Fix get_attn_mask: compute in float32 on CPU, cache on GPU.

    Works for both CPU and GPU inference. On CUDA, prefer patch_attn_mask_gpu
    for faster cold-start (first inference).
    """
    device = next(model.encoder.parameters()).device

    def _fast_get_attn_mask(self, height, width, dtype):
        if self.shift_size == 0:
            return None
        key = (height, width, dtype)
        if not hasattr(self, "_mask_cache"):
            self._mask_cache = {}
        if key not in self._mask_cache:
            ws, ss = self.window_size, self.shift_size
            # float32 on CPU — fast on all hardware; no AVX-512 FP16 required
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
            # dtype conversion + GPU transfer done once; .to(device) in forward() is a no-op
            self._mask_cache[key] = mask.to(device=device, dtype=dtype)
        return self._mask_cache[key]

    for stage in model.encoder.encoder.layers:
        for block in stage.blocks:
            block.get_attn_mask = types.MethodType(_fast_get_attn_mask, block)


def patch_attn_mask_gpu(model) -> None:
    """Fix get_attn_mask: create mask directly on GPU, cache on GPU.

    ~75x faster cold-start than patch_attn_mask (no CPU compute, no H2D transfer).
    Requires CUDA. Use patch_attn_mask for CPU-only inference.
    """
    device = next(model.encoder.parameters()).device
    if device.type != "cuda":
        raise ValueError(
            "patch_attn_mask_gpu requires a CUDA device; use patch_attn_mask for CPU inference"
        )

    def _fast_get_attn_mask_gpu(self, height, width, dtype):
        if self.shift_size == 0:
            return None
        key = (height, width, dtype)
        if not hasattr(self, "_mask_cache"):
            self._mask_cache = {}
        if key not in self._mask_cache:
            ws, ss = self.window_size, self.shift_size
            # float32 on GPU — 9 kernel launches for slice assignments (~0.3ms total)
            # vs ~23ms for CPU float32 + H2D transfer (stage-0 mask)
            img_mask = torch.zeros((1, height, width, 1), device=device)
            cnt = 0
            for h in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                for w in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mw = img_mask.view(1, height // ws, ws, width // ws, ws, 1)
            mw = mw.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws)
            mask = mw.unsqueeze(1) - mw.unsqueeze(2)
            mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
            self._mask_cache[key] = mask.to(dtype=dtype)  # GPU dtype cast, no transfer
        return self._mask_cache[key]

    for stage in model.encoder.encoder.layers:
        for block in stage.blocks:
            block.get_attn_mask = types.MethodType(_fast_get_attn_mask_gpu, block)
