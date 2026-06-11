"""Cyclic-shift attention mask caching for DonutSwin.

DonutSwin recomputes the cyclic-shift window mask on every forward pass, which
is wasteful — the mask depends only on (height, width, dtype) and never changes
between calls. This module replaces the per-call computation with a cached
version: the mask is computed once per (height, width, dtype) triplet and reused.

Two implementations are provided internally; apply_mask_cache() selects
automatically based on the model device:
  - CPU / GPU-transfer: compute in float32 on CPU, transfer once to device.
  - GPU-direct: compute entirely on GPU (no host compute, no H2D transfer).
    ~75x faster cold-start on CUDA; meaningless on CPU.

The cached function keeps the (height, width, dtype, device) signature that
transformers v5 uses when calling get_attn_mask.
"""

import types

import torch


def _make_cpu_variant(device):
    def _get_attn_mask_cached(self, height, width, dtype, device=device):
        if self.shift_size == 0:
            return None
        key = (height, width, dtype)
        if not hasattr(self, "_mask_cache"):
            self._mask_cache = {}
        if key not in self._mask_cache:
            ws, ss = self.window_size, self.shift_size
            # float32 on CPU — compatible with all hardware
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
            # Transfer to device once; subsequent calls hit the cache
            self._mask_cache[key] = mask.to(device=device, dtype=dtype)
        return self._mask_cache[key]

    return _get_attn_mask_cached


def _make_gpu_variant(device):
    def _get_attn_mask_cached_gpu(self, height, width, dtype, device=device):
        if self.shift_size == 0:
            return None
        key = (height, width, dtype)
        if not hasattr(self, "_mask_cache"):
            self._mask_cache = {}
        if key not in self._mask_cache:
            ws, ss = self.window_size, self.shift_size
            # Compute directly on GPU — 9 kernel launches (~0.3 ms total)
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
            self._mask_cache[key] = mask.to(dtype=dtype)
        return self._mask_cache[key]

    return _get_attn_mask_cached_gpu


def apply_mask_cache(model) -> None:
    """Cache the cyclic-shift attention mask on every shifted Swin block.

    Replaces the default per-call mask recomputation with a cached version keyed
    by (height, width, dtype). Auto-selects the GPU-direct variant on CUDA and
    the CPU-float32 variant elsewhere. Idempotent.
    """
    device = next(model.encoder.parameters()).device
    make_fn = _make_gpu_variant if device.type == "cuda" else _make_cpu_variant
    cached_fn = make_fn(device)

    for stage in model.encoder.encoder.layers:
        for block in stage.blocks:
            if hasattr(block, "_mask_cache_applied"):
                continue
            block.get_attn_mask = types.MethodType(cached_fn, block)
            block._mask_cache_applied = True


def revert_mask_cache(model) -> None:
    """Undo apply_mask_cache: restore the class get_attn_mask and drop caches.

    Deleting the per-instance attributes makes each block fall back to the
    original get_attn_mask defined on the class. No-op if not applied.
    """
    for stage in model.encoder.encoder.layers:
        for block in stage.blocks:
            if not hasattr(block, "_mask_cache_applied"):
                continue
            # Instance attrs shadow the class method; deleting restores the original.
            del block.get_attn_mask
            del block._mask_cache_applied
            if hasattr(block, "_mask_cache"):
                del block._mask_cache


def check_mask_cache(model) -> None:
    """Assert mask caching is active on every shifted Swin block."""
    for i, stage in enumerate(model.encoder.encoder.layers):
        for j, block in enumerate(stage.blocks):
            if block.shift_size == 0:
                continue
            assert getattr(block, "_mask_cache_applied", False), (
                f"Stage {i} block {j} (shift_size={block.shift_size}) "
                "does not have mask caching applied — call apply_mask_cache first"
            )
