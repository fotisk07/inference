"""Time a single encoder+decoder forward pass to debug slow inference."""

from __future__ import annotations

import argparse
import time

import torch
from datasets import load_dataset

from benchmark.model import ModelBundle

MODEL_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"
TASK_PROMPT = "<s_cord-v2>"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=1, help="warmup runs before timing")
    args = parser.parse_args()

    print(f"Device : {args.device}")
    print(f"Model  : {args.model}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print()

    print("Loading model...")
    bundle = time_section("model load", lambda: ModelBundle.load(args.model, args.device, TASK_PROMPT))

    print("\nLoading one sample image...")
    ds = load_dataset("naver-clova-ix/cord-v2", split="test", streaming=True)
    sample = next(iter(ds))
    image = sample["image"]

    prep = bundle.preprocess([image])
    print(f"  pixel_values shape : {prep.pixel_values.shape}")
    print(f"  processed size     : {prep.processed_image_width}x{prep.processed_image_height}")
    print()

    def run_encode():
        return bundle.encode(prep.pixel_values)

    def run_decode(enc_out):
        return bundle.decode(prep.pixel_values, enc_out, prep.decoder_input_ids)

    # Warmup
    if args.warmup > 0:
        print(f"Warming up ({args.warmup} run(s))...")
        for _ in range(args.warmup):
            enc = run_encode()
            run_decode(enc)
        print()

    # Timed run
    print("Timed forward pass:")
    enc_out = time_section("encode", run_encode)
    sequences = time_section("decode", lambda: run_decode(enc_out))

    actual_lens, max_len, sum_len, _ = bundle.count_tokens(sequences, prep.decoder_input_ids.shape[1])
    print(f"\n  generated tokens   : {actual_lens[0]}")

    # Also time preprocessing separately
    print("\nPreprocessing timing:")
    time_section("preprocess", lambda: bundle.preprocess([image]))


if __name__ == "__main__":
    main()
