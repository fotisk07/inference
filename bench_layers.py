"""Donut layer-level profiler.

Times each high-level component of the encoder and decoder using PyTorch forward
hooks backed by CUDA events. CUDA events have near-zero CPU overhead per call —
critical here because the decoder hooks fire once per generated token.

Encoder breakdown: patch embedding + each Swin stage (once per forward pass).
Decoder breakdown: embed_tokens + each MBart decoder layer + lm_head (once per
token). The decoder total reported is the sum of hooked module times only; it
excludes CPU sampling overhead between tokens (see bench.py for wall-clock total).

Requires CUDA. Use bench.py / diagnose.py for CPU profiling.

Usage:
    uv run bench_layers.py [--warmup 2] [--save layers.json]
    uv run bench_layers.py --no-patch --image test_data/test_data.jpg
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
import time

import torch
import transformers
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from patches import patch_attn_mask_gpu

MODEL_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"
TASK_PROMPT = "<s_cord-v2>"
TEST_IMAGE = "test_data/test_data.jpg"


def cuda_sync():
    torch.cuda.synchronize()


class CudaTimer:
    """Wall-clock timer with cuda sync."""

    def __init__(self):
        self._start = None

    def start(self):
        cuda_sync()
        self._start = time.perf_counter()

    def stop(self) -> float:
        cuda_sync()
        return (time.perf_counter() - self._start) * 1000.0


# ── Layer timer via CUDA events ───────────────────────────────────────────────


class LayerTimer:
    """Collects per-call GPU timing via CUDA event pairs attached as hooks."""

    def __init__(self, name: str):
        self.name = name
        self.events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._start: torch.cuda.Event | None = None

    def pre_hook(self, module, input):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._start = e

    def post_hook(self, module, input, output):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.events.append((self._start, e))

    def elapsed_no_sync(self) -> list[float]:
        """Read elapsed times. Caller must have called torch.cuda.synchronize() first."""
        return [s.elapsed_time(end) for s, end in self.events]


def register_timers(model) -> tuple[list[LayerTimer], list]:
    """Attach LayerTimers to encoder stages and decoder layers. Returns (timers, handles)."""
    timers: list[LayerTimer] = []
    handles: list = []

    def attach(module, name: str) -> LayerTimer:
        t = LayerTimer(name)
        handles.append(module.register_forward_pre_hook(t.pre_hook))
        handles.append(module.register_forward_hook(t.post_hook))
        timers.append(t)
        return t

    # Encoder
    attach(model.encoder.embeddings, "enc:patch_embed")
    for i, stage in enumerate(model.encoder.encoder.layers):
        attach(stage, f"enc:stage{i}")

    # Decoder
    attach(model.decoder.model.decoder.embed_tokens, "dec:embed_tokens")
    for i, layer in enumerate(model.decoder.model.decoder.layers):
        attach(layer, f"dec:layer{i}")
    attach(model.decoder.lm_head, "dec:lm_head")

    return timers, handles


def remove_handles(handles: list) -> None:
    for h in handles:
        h.remove()


# ── Inference helpers ─────────────────────────────────────────────────────────


def run_pass(model, processor, pixel_values, decoder_input_ids):
    """Full encode+decode pass. Returns (enc_out, seqs)."""
    with torch.no_grad():
        enc_out = model.encoder(pixel_values, return_dict=True)
        seqs = model.generate(
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
    return enc_out, seqs


# ── Reporting ─────────────────────────────────────────────────────────────────


def _system_info() -> dict:
    info: dict = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = str(torch.backends.cudnn.version())
    return info


def print_encoder(timers_by_name: dict[str, list[float]], swin):
    enc_keys = [k for k in timers_by_name if k.startswith("enc:")]
    total = sum(timers_by_name[k][0] for k in enc_keys)
    col = 22

    print("Encoder breakdown (1 pass):")
    ms = timers_by_name["enc:patch_embed"][0]
    print(f"  {'patch_embed':<{col}} : {ms:7.1f} ms")

    for i, stage in enumerate(swin.encoder.layers):
        key = f"enc:stage{i}"
        ms = timers_by_name[key][0]
        n = len(stage.blocks)
        print(
            f"  {f'stage {i}  ({n:2d} blocks)':<{col}} : {ms:7.1f} ms    {ms / n:5.1f} ms/block"
        )

    print(f"  {'─' * (col + 18)}")
    print(f"  {'total encoder':<{col}} : {total:7.1f} ms")
    print()


def print_decoder(timers_by_name: dict[str, list[float]], n_tokens: int):
    dec_keys = [k for k in timers_by_name if k.startswith("dec:")]
    totals = {k: sum(timers_by_name[k]) for k in dec_keys}
    grand_total = sum(totals.values())
    col = 22

    print(f"Decoder breakdown (N={n_tokens} tokens, hooked module time only):")
    for key in dec_keys:
        samples = timers_by_name[key]
        total_ms = totals[key]
        per_tok = total_ms / max(len(samples), 1)
        pct = 100.0 * total_ms / grand_total if grand_total > 0 else 0.0
        label = key.removeprefix("dec:")
        print(
            f"  {label:<{col}} : {per_tok:6.2f} ms/tok  {total_ms:8.1f} ms total  {pct:5.1f}%"
        )

    print(f"  {'─' * (col + 38)}")
    per_tok_total = grand_total / n_tokens if n_tokens > 0 else 0.0
    print(
        f"  {'total (hooked)':<{col}} : {per_tok_total:6.2f} ms/tok  {grand_total:8.1f} ms total"
        f"  (excl. CPU sampling overhead)"
    )
    print()


def save_stats(
    path: str,
    args,
    input_shape: list[int],
    n_tokens: int,
    timers_by_name: dict[str, list[float]],
    swin,
) -> None:
    enc_keys = sorted(k for k in timers_by_name if k.startswith("enc:"))
    dec_keys = [k for k in timers_by_name if k.startswith("dec:")]

    enc_data: dict = {}
    for key in enc_keys:
        field = key.removeprefix("enc:").replace(":", "_")
        enc_data[f"{field}_ms"] = round(timers_by_name[key][0], 3)
    enc_data["total_ms"] = round(sum(timers_by_name[k][0] for k in enc_keys), 3)

    dec_layers: dict = {}
    dec_total = 0.0
    for key in dec_keys:
        samples = timers_by_name[key]
        total_ms = sum(samples)
        dec_total += total_ms
        per_tok = total_ms / max(len(samples), 1)
        dec_layers[key.removeprefix("dec:")] = {
            "per_token_ms_mean": round(per_tok, 4),
            "total_ms": round(total_ms, 3),
            "pct": round(100.0 * total_ms / dec_total, 2) if dec_total > 0 else 0.0,
            "samples": [round(v, 4) for v in samples],
        }
    # Recompute pct with final dec_total
    for v in dec_layers.values():
        v["pct"] = round(100.0 * v["total_ms"] / dec_total, 2) if dec_total > 0 else 0.0

    data = {
        "schema_version": 1,
        "script": "bench_layers.py",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "args": {
            "model": args.model,
            "device": args.device,
            "warmup": args.warmup,
            "no_patch": args.no_patch,
            "image": args.image,
        },
        "system": _system_info(),
        "input_shape": input_shape,
        "n_tokens": n_tokens,
        "encoder": enc_data,
        "decoder": {
            "note": "sum of hooked module times; excludes CPU sampling overhead between tokens",
            "total_ms": round(dec_total, 3),
            "layers": dec_layers,
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  stats saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Donut layer-level profiler (CUDA only)"
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--no-patch", action="store_true")
    parser.add_argument("--image", default=TEST_IMAGE)
    parser.add_argument("--save", default=None, metavar="PATH")
    args = parser.parse_args()

    dev = args.device
    if dev != "cuda":
        sys.exit(
            "bench_layers.py requires CUDA (CUDA events needed for per-layer timing)"
        )

    # ── Header ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Donut layer profiler")
    print("=" * 60)
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  Transformers: {transformers.__version__}")
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  CUDA        : {torch.version.cuda}")
    print(f"  cuDNN       : {torch.backends.cudnn.version()}")
    print(f"  Compute cap : {torch.cuda.get_device_capability()}")
    print(f"  Device      : {dev}")

    # ── Load model ───────────────────────────────────────────────────────────
    print("\nLoading model...")
    processor = DonutProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    )
    model.to(dev).eval()

    enc_dtype = next(model.encoder.parameters()).dtype
    print(f"  dtype       : {enc_dtype}")

    if args.no_patch:
        print("  patch       : DISABLED (--no-patch)")
    else:
        patch_attn_mask_gpu(model)
        print("  patch       : applied (gpu direct)")

    # ── Preprocess ───────────────────────────────────────────────────────────
    image = Image.open(args.image).convert("RGB")
    pixel_values = (
        processor(image, return_tensors="pt").pixel_values.to(dev).to(model.dtype)
    )
    decoder_input_ids = processor.tokenizer(
        TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(dev)
    print(f"\n  input shape : {list(pixel_values.shape)}\n")

    # ── Warmup ───────────────────────────────────────────────────────────────
    print(f"Warmup ({args.warmup} run(s))...")
    for _ in range(args.warmup):
        run_pass(model, processor, pixel_values, decoder_input_ids)
    cuda_sync()

    # ── Register hooks and run profiled pass ─────────────────────────────────
    print("Running profiled pass...")
    timers, handles = register_timers(model)
    _, seqs = run_pass(model, processor, pixel_values, decoder_input_ids)

    # Single synchronize, then read all CUDA event times
    torch.cuda.synchronize()
    timers_by_name: dict[str, list[float]] = {
        t.name: t.elapsed_no_sync() for t in timers
    }
    remove_handles(handles)

    # Token count
    prompt_len = decoder_input_ids.shape[1]
    row = seqs[0, prompt_len:]
    eos = (row == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    n_tokens = int(eos[0].item()) + 1 if len(eos) > 0 else len(row)

    print()

    # ── Print results ─────────────────────────────────────────────────────────
    print_encoder(timers_by_name, model.encoder)
    print_decoder(timers_by_name, n_tokens)

    if args.save:
        save_stats(
            args.save,
            args,
            list(pixel_values.shape),
            n_tokens,
            timers_by_name,
            model.encoder,
        )


if __name__ == "__main__":
    main()
