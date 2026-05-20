"""Model patches for Donut inference.

Import and call patch_attn_mask(model) after loading any DonutSwinModel-based model.
The patch fixes two bugs in DonutSwinLayer.get_attn_mask (transformers 4.37.2):
  1. Float16 arithmetic on CPU — falls back to software emulation on CPUs without
     AVX-512 FP16, making masked_fill ~500x slower than on GPU.
  2. Mask recomputed on every forward pass — it is a deterministic function of
     (window_size, shift_size, height, width) and should be computed once.

See H100_swin_debug_report.md for the full investigation.
"""

import types

import torch


def patch_attn_mask(model) -> None:
    """Replace get_attn_mask on every shifted Swin block with a cached float32 version.

    Safe to call multiple times — subsequent calls overwrite earlier patches.
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
