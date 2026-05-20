"""Time a single encoder+decoder forward pass to debug slow inference.

Standalone script — no local package imports required.
Dependencies: torch, transformers, datasets, Pillow
"""

from __future__ import annotations

import argparse
import time

import sys

import torch
import transformers
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

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

    print(f"Python        : {sys.version.split()[0]}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Transformers  : {transformers.__version__}")
    print(f"Device        : {args.device}")
    print(f"Model         : {args.model}")
    if torch.cuda.is_available():
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        print(f"CUDA          : {torch.version.cuda}")
    print()

    print("Loading model...")
    t0 = time.perf_counter()
    processor = DonutProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.to(args.device)
    model.eval()
    cuda_sync()
    print(f"  {'model load':<30} {(time.perf_counter()-t0)*1000:8.1f} ms")
    print(f"  encoder dtype         : {next(model.encoder.parameters()).dtype}")
    print(f"  decoder dtype         : {next(model.decoder.parameters()).dtype}\n")

    print("Loading one sample image...")
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


if __name__ == "__main__":
    main()
