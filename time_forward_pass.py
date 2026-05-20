"""Time a single encoder+decoder forward pass to debug slow inference.

Standalone script — no local package imports required.
Dependencies: torch, transformers, Pillow
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import transformers
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

MODEL_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"
TASK_PROMPT = "<s_cord-v2>"


def patch_attn_mask(model):
    """Fix DonutSwinLayer.get_attn_mask:
    - Computes in float32 on CPU (no native fp16 ALU on most CPUs → 500x slower)
    - Caches the result on GPU per (height, width, dtype) so subsequent calls
      are a free dict lookup — no recomputation, no CPU→GPU copy, no dtype cast.
    """
    import types
    device = next(model.encoder.parameters()).device

    def _fast_get_attn_mask(self, height, width, dtype):
        if self.shift_size == 0:
            return None
        key = (height, width, dtype)  # dtype in key so we cache the final form
        if not hasattr(self, "_mask_cache"):
            self._mask_cache = {}
        if key not in self._mask_cache:
            ws, ss = self.window_size, self.shift_size
            img_mask = torch.zeros((1, height, width, 1))  # float32 on CPU — fast
            cnt = 0
            for h in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                for w in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mw = img_mask.view(1, height // ws, ws, width // ws, ws, 1)
            mw = mw.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws)
            mask = mw.unsqueeze(1) - mw.unsqueeze(2)
            mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
            # dtype conversion + GPU transfer done once; GPU is fast at both
            self._mask_cache[key] = mask.to(device=device, dtype=dtype)
        return self._mask_cache[key]  # already on GPU, .to(device) in forward() is a no-op

    for stage in model.encoder.encoder.layers:
        for block in stage.blocks:
            block.get_attn_mask = types.MethodType(_fast_get_attn_mask, block)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_section(label: str, fn):
    cuda_sync()
    t0 = time.perf_counter()
    result = fn()
    cuda_sync()
    elapsed = time.perf_counter() - t0
    print(f"  {label:<30} {elapsed*1000:8.1f} ms")
    return result


def matmul_bench(device, dtype, shape_a, shape_b, iters=50):
    """Time a matmul with CUDA events. shape_a/shape_b are full tensor shapes."""
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


def _window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def _window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def step(label, fn, indent=4):
    cuda_sync()
    t0 = time.perf_counter()
    result = fn()
    cuda_sync()
    print(f"  {' '*indent}{label:<38} {(time.perf_counter()-t0)*1000:8.2f} ms")
    return result


def diagnose_block(swin, pixel_values):
    """Time every individual operation inside stage-0 block-1 (the shifted block)."""
    print("\nBlock-level diagnostic (stage 0, block 1 — shifted):")
    with torch.no_grad():
        hidden_states, output_dims = swin.embeddings(pixel_values)
        stage = swin.encoder.layers[0]
        # advance through block 0 silently to get correct input for block 1
        hidden_states = stage.blocks[0](hidden_states, output_dims)[0]

        block = stage.blocks[1]
        height, width = output_dims
        B, _, C = hidden_states.size()
        ws = block.window_size
        ss = block.shift_size
        shortcut = hidden_states

        hs = step("layernorm_before", lambda: block.layernorm_before(hidden_states))
        hs = step("view to (B,H,W,C)", lambda: hs.view(B, height, width, C))
        hs, pad_values = block.maybe_pad(hs, height, width)
        _, Hp, Wp, _ = hs.shape

        # CPU-side mask creation (not a GPU op)
        t0 = time.perf_counter()
        attn_mask = block.get_attn_mask(Hp, Wp, dtype=hs.dtype)
        print(f"      {'get_attn_mask (CPU)':<38} {(time.perf_counter()-t0)*1000:8.2f} ms")

        if attn_mask is not None:
            print(f"        mask shape: {attn_mask.shape}, size: {attn_mask.numel()*2/1024:.0f} KB")
            attn_mask = step("attn_mask .to(device)", lambda: attn_mask.to(hs.device))

        hs = step("torch.roll", lambda: torch.roll(hs, shifts=(-ss, -ss), dims=(1, 2)))
        hs_w = step("window_partition", lambda: _window_partition(hs, ws).view(-1, ws*ws, C))
        attn_out = step("self.attention", lambda: block.attention(hs_w, attn_mask, None))
        aw = attn_out[0]
        aw = step("window_reverse", lambda: _window_reverse(aw.view(-1, ws, ws, C), ws, Hp, Wp))
        aw = step("torch.roll (unshift)", lambda: torch.roll(aw, shifts=(ss, ss), dims=(1, 2)))
        aw = step("view + drop_path + add", lambda: shortcut + block.drop_path(aw[:, :height, :width].contiguous().view(B, height*width, C)))
        lo = step("layernorm_after", lambda: block.layernorm_after(aw))
        lo = step("intermediate (FFN)", lambda: block.intermediate(lo))
        step("output + residual", lambda: aw + block.output(lo))


def encode_per_stage(swin, pixel_values):
    """Run encoder stage-by-stage and print timing for each Swin stage."""
    with torch.no_grad():
        cuda_sync()
        t0 = time.perf_counter()
        hidden_states, output_dims = swin.embeddings(pixel_values)
        cuda_sync()
        print(f"  {'patch embed':<28} {(time.perf_counter()-t0)*1000:8.1f} ms")

        input_dimensions = output_dims
        for i, stage in enumerate(swin.encoder.layers):
            cuda_sync()
            t0 = time.perf_counter()
            stage_outputs = stage(hidden_states, input_dimensions)
            cuda_sync()
            elapsed = (time.perf_counter() - t0) * 1000
            hidden_states = stage_outputs[0]
            output_dimensions = stage_outputs[2]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            n_blocks = len(stage.blocks)
            print(f"  {'stage '+str(i)+' ('+str(n_blocks)+' blocks)':<28} {elapsed:8.1f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--compile", action="store_true", help="torch.compile the encoder")
    parser.add_argument("--cudnn-benchmark", action="store_true", help="enable cudnn.benchmark autotuning")
    args = parser.parse_args()

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    print(f"Python        : {sys.version.split()[0]}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Transformers  : {transformers.__version__}")
    print(f"Device        : {args.device}")
    print(f"Model         : {args.model}")
    if torch.cuda.is_available():
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        print(f"CUDA          : {torch.version.cuda}")
        print(f"cuDNN         : {torch.backends.cudnn.version()}")
        print(f"Compute cap   : {torch.cuda.get_device_capability()}")
        print(f"cudnn.benchmark : {torch.backends.cudnn.benchmark}")
    print()

    # Matmul microbenchmarks — large (compute-bound) vs small-batched (Swin-like)
    if torch.cuda.is_available():
        dtype = torch.float16
        print("Matmul microbenchmarks:")
        # Large — should be fast everywhere
        ms_large = matmul_bench(args.device, dtype, (1024, 1024), (1024, 1024))
        print(f"  {'large  1024x1024x1024':<32} {ms_large:7.3f} ms")
        # Swin stage-0 attention: 1200 windows, 4 heads, 64 tokens, head_dim=32
        ms_s0 = matmul_bench(args.device, dtype, (1200, 4, 64, 32), (1200, 4, 32, 64))
        print(f"  {'swin-s0 (1200,4,64,32)@(32,64)':<32} {ms_s0:7.3f} ms")
        # Swin stage-2 attention: ~75 windows, 16 heads, 64 tokens, head_dim=32
        ms_s2 = matmul_bench(args.device, dtype, (75, 16, 64, 32), (75, 16, 32, 64))
        print(f"  {'swin-s2  (75,16,64,32)@(32,64)':<32} {ms_s2:7.3f} ms")
        print()

    print("Loading model...")
    t0 = time.perf_counter()
    processor = DonutProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.to(args.device)
    model.eval()
    cuda_sync()
    print(f"  {'model load':<30} {(time.perf_counter()-t0)*1000:8.1f} ms")
    enc_dtype = next(model.encoder.parameters()).dtype
    dec_dtype = next(model.decoder.parameters()).dtype
    print(f"  encoder dtype         : {enc_dtype}")
    print(f"  decoder dtype         : {dec_dtype}")
    if enc_dtype != torch.bfloat16:
        print(f"  WARNING: expected bfloat16, got {enc_dtype}")

    patch_attn_mask(model)
    print("  attn_mask patch applied (float32 + cached)\n")

    # Keep original swin for per-stage timing regardless of compile
    orig_swin = model.encoder

    if args.compile:
        print("Compiling encoder with torch.compile ...")
        model.encoder = torch.compile(model.encoder, dynamic=True)
        print()

    print("Loading image...")
    image = Image.open("test_data/test_data.jpg").convert("RGB")

    def preprocess():
        pv = processor(image, return_tensors="pt").pixel_values.to(args.device).to(model.dtype)
        ids = processor.tokenizer(TASK_PROMPT, add_special_tokens=False, return_tensors="pt").input_ids.to(args.device)
        return pv, ids

    pixel_values, decoder_input_ids = preprocess()
    print(f"  pixel_values shape : {pixel_values.shape}\n")

    def encode():
        with torch.no_grad():
            return model.encoder(pixel_values, return_dict=True)

    def decode(encoder_outputs):
        with torch.no_grad():
            return model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                max_length=model.decoder.config.max_position_embeddings,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            ).sequences

    if args.warmup > 0:
        print(f"Warming up ({args.warmup} run(s))...")
        for _ in range(args.warmup):
            decode(encode())
        print()

    print("Timed forward pass:")
    time_section("preprocess", preprocess)
    enc_out = time_section("encode", encode)
    sequences = time_section("decode", lambda: decode(enc_out))

    prompt_len = decoder_input_ids.shape[1]
    row = sequences[0, prompt_len:]
    eos_positions = (row == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    n_tokens = int(eos_positions[0].item()) + 1 if len(eos_positions) > 0 else len(row)
    print(f"\n  generated tokens   : {n_tokens}")

    print("\nPer-stage encoder breakdown:")
    encode_per_stage(orig_swin, pixel_values)

    diagnose_block(orig_swin, pixel_values)


if __name__ == "__main__":
    main()
