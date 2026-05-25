"""Donut diagnostic tool — deep timing breakdown for the encoder.

Use this when you need to understand WHERE time goes inside the model,
not just the total. Run after bench.py establishes the high-level numbers.

Usage:
    uv run diagnose.py                          # full diagnostic, patch applied
    uv run diagnose.py --no-patch               # reproduce original slow behaviour
    uv run diagnose.py --matmul-bench           # also run GPU health check
    uv run diagnose.py --no-stage-breakdown     # skip per-stage table
    uv run diagnose.py --no-block-diagnose      # skip per-op block table
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import transformers
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from patches import patch_attn_mask

MODEL_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"
TASK_PROMPT = "<s_cord-v2>"
TEST_IMAGE = "test_data/test_data.jpg"


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(label: str, fn, indent: int = 2):
    cuda_sync()
    t0 = time.perf_counter()
    result = fn()
    cuda_sync()
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {' ' * indent}{label:<38} {ms:8.2f} ms")
    return result


# ── GPU health check ─────────────────────────────────────────────────────────

def _matmul_bench(device, dtype, shape_a, shape_b, iters=50) -> float:
    a = torch.randn(*shape_a, device=device, dtype=dtype)
    b = torch.randn(*shape_b, device=device, dtype=dtype)
    for _ in range(5):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_matmul_bench(device):
    dtype = torch.float16
    print("Matmul microbenchmarks (CUDA events, 50 iters):")
    ms = _matmul_bench(device, dtype, (1024, 1024), (1024, 1024))
    print(f"  {'large  1024×1024×1024':<38} {ms:7.3f} ms")
    ms = _matmul_bench(device, dtype, (1200, 4, 64, 32), (1200, 4, 32, 64))
    print(f"  {'swin-s0 (1200,4,64,32)@(32,64)':<38} {ms:7.3f} ms")
    ms = _matmul_bench(device, dtype, (75, 16, 64, 32), (75, 16, 32, 64))
    print(f"  {'swin-s2  (75,16,64,32)@(32,64)':<38} {ms:7.3f} ms")
    print()


# ── Per-stage encoder timing ──────────────────────────────────────────────────

def run_stage_breakdown(swin, pixel_values):
    print("Per-stage encoder breakdown:")
    with torch.no_grad():
        cuda_sync()
        t0 = time.perf_counter()
        hidden_states, output_dims = swin.embeddings(pixel_values)
        cuda_sync()
        print(f"  {'patch embed':<38} {(time.perf_counter()-t0)*1000:8.1f} ms")

        input_dims = output_dims
        for i, stage in enumerate(swin.encoder.layers):
            cuda_sync()
            t0 = time.perf_counter()
            stage_out = stage(hidden_states, input_dims)
            cuda_sync()
            ms = (time.perf_counter() - t0) * 1000
            hidden_states = stage_out[0]
            out_dims = stage_out[2]
            input_dims = (out_dims[-2], out_dims[-1])
            n = len(stage.blocks)
            print(f"  {'stage '+str(i)+' ('+str(n)+' blocks)':<38} {ms:8.1f} ms   ({ms/n:.0f} ms/block)")
    print()


# ── Per-operation breakdown inside one block ──────────────────────────────────

def _window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def _window_reverse(windows, ws, H, W):
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def run_block_diagnose(swin, pixel_values):
    print("Block-level diagnostic — stage 0, block 1 (shifted window):")
    with torch.no_grad():
        hidden_states, output_dims = swin.embeddings(pixel_values)
        stage = swin.encoder.layers[0]
        # run block 0 silently to get the correct input for block 1
        hidden_states = stage.blocks[0](hidden_states, output_dims)[0]

        block = stage.blocks[1]
        height, width = output_dims
        B, _, C = hidden_states.size()
        ws, ss = block.window_size, block.shift_size
        shortcut = hidden_states

        hs = timed("layernorm_before", lambda: block.layernorm_before(hidden_states))
        hs = timed("view to (B,H,W,C)", lambda: hs.view(B, height, width, C))
        hs, _ = block.maybe_pad(hs, height, width)
        _, Hp, Wp, _ = hs.shape

        # get_attn_mask runs on CPU — measure with wall clock, not cuda_sync
        t0 = time.perf_counter()
        attn_mask = block.get_attn_mask(Hp, Wp, dtype=hs.dtype)
        cpu_ms = (time.perf_counter() - t0) * 1000
        print(f"    {'get_attn_mask (CPU)':<38} {cpu_ms:8.2f} ms")

        if attn_mask is not None:
            kb = attn_mask.numel() * 2 / 1024
            print(f"      mask shape: {list(attn_mask.shape)}, size: {kb:.0f} KB")
            attn_mask = timed("attn_mask .to(device)", lambda: attn_mask.to(hs.device))

        hs = timed("torch.roll", lambda: torch.roll(hs, shifts=(-ss, -ss), dims=(1, 2)))
        hs_w = timed("window_partition", lambda: _window_partition(hs, ws).view(-1, ws * ws, C))
        attn_out = timed("self.attention", lambda: block.attention(hs_w, attn_mask, None))
        aw = attn_out[0]
        aw = timed("window_reverse", lambda: _window_reverse(aw.view(-1, ws, ws, C), ws, Hp, Wp))
        aw = timed("torch.roll (unshift)", lambda: torch.roll(aw, shifts=(ss, ss), dims=(1, 2)))
        aw = timed(
            "view + drop_path + add",
            lambda: shortcut + block.drop_path(aw[:, :height, :width].contiguous().view(B, height * width, C)),
        )
        lo = timed("layernorm_after", lambda: block.layernorm_after(aw))
        lo = timed("intermediate (FFN)", lambda: block.intermediate(lo))
        timed("output + residual", lambda: aw + block.output(lo))
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Donut encoder diagnostic")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-patch", action="store_true", help="skip attn_mask patch — reproduces slow behaviour")
    parser.add_argument("--matmul-bench", action="store_true", help="run GPU matmul health check")
    parser.add_argument("--no-stage-breakdown", action="store_true", help="skip per-stage encoder timing")
    parser.add_argument("--no-block-diagnose", action="store_true", help="skip per-op block diagnostic")
    args = parser.parse_args()

    dev = args.device

    # ── Header ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Donut encoder diagnostic")
    print("=" * 60)
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  Transformers: {transformers.__version__}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA        : {torch.version.cuda}")
        print(f"  cuDNN       : {torch.backends.cudnn.version()}")
        print(f"  Compute cap : {torch.cuda.get_device_capability()}")
    print(f"  Device      : {dev}")
    print()

    if args.matmul_bench and torch.cuda.is_available():
        run_matmul_bench(dev)

    # ── Load model ───────────────────────────────────────────────────────────
    print("Loading model...")
    processor = DonutProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.to(dev).eval()

    enc_dtype = next(model.encoder.parameters()).dtype
    print(f"  dtype       : {enc_dtype}")

    if args.no_patch:
        print("  patch       : DISABLED (--no-patch)")
    else:
        patch_attn_mask(model)
        print("  patch       : applied")
    print()

    # ── Preprocess ───────────────────────────────────────────────────────────
    image = Image.open(TEST_IMAGE).convert("RGB")
    pixel_values = (
        processor(image, return_tensors="pt")
        .pixel_values.to(dev)
        .to(model.dtype)
    )
    decoder_input_ids = (
        processor.tokenizer(TASK_PROMPT, add_special_tokens=False, return_tensors="pt")
        .input_ids.to(dev)
    )
    print(f"Input shape: {list(pixel_values.shape)}\n")

    # ── Single timed pass (always shown) ─────────────────────────────────────
    print("Timed forward pass (1 run):")

    def encode():
        with torch.no_grad():
            return model.encoder(pixel_values, return_dict=True)

    def decode(enc_out):
        with torch.no_grad():
            return model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=enc_out,
                max_length=model.decoder.config.max_position_embeddings,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            ).sequences

    enc_out = timed("encode", encode, indent=0)
    seqs = timed("decode", lambda: decode(enc_out), indent=0)

    prompt_len = decoder_input_ids.shape[1]
    row = seqs[0, prompt_len:]
    eos = (row == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    n_tokens = int(eos[0].item()) + 1 if len(eos) > 0 else len(row)
    print(f"  generated tokens: {n_tokens}\n")

    # ── Deep diagnostics ─────────────────────────────────────────────────────
    swin = model.encoder  # DonutSwinModel

    if not args.no_stage_breakdown:
        run_stage_breakdown(swin, pixel_values)

    if not args.no_block_diagnose:
        run_block_diagnose(swin, pixel_values)


if __name__ == "__main__":
    main()
