"""Donut dataset benchmark — high-level component timing.

Measures preprocessing, encoder, and decoder latency over a pool of real images
Prints a 5-line summary; saves full per-run raw data to JSON with --save.

Usage:
    uv run scripts/bench_dataset.py [--pool 50] [--runs 50] [--batch-size 1]
    uv run scripts/bench_dataset.py --dataset naver-clova-ix/cord-v2 --pool 50
    uv run scripts/bench_dataset.py --image-dir /path/to/images --pool 20 --runs 50
    uv run scripts/bench_dataset.py --batch-size 4 --max-new-tokens 100 --save out.json
"""

import datetime
import statistics
import time

import torch
from PIL import Image
from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict

from inference.constants import DEFAULT_DATASET, TASK_PROMPT
from inference.data import load_pool, sample_batch
from inference.model import apply_patch, load_model
from inference.saving import atomic_save_json
from inference.settings import BenchSettings
from inference.stats import stat, system_info
from inference.timing import CudaTimer


class Settings(BenchSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True, env_prefix="BENCH_", cli_prog_name="bench_dataset"
    )
    runs: int = Field(default=50, description="Number of measurement runs")
    pool: int = Field(default=20, description="Size of image pool to sample from")
    batch_size: int = Field(default=1, description="Batch size per run")
    max_new_tokens: int | None = Field(
        default=None, description="Cap decoder generation length (default: uncapped)"
    )
    dataset: str | None = Field(
        default=DEFAULT_DATASET, description="HuggingFace dataset ID"
    )
    image_dir: str | None = Field(
        default=None,
        description="Local image directory (mutually exclusive with dataset)",
    )
    dataset_split: str = Field(default="test")
    image_column: str = Field(default="image")

    @model_validator(mode="after")
    def check_exclusive_source(self):
        if (
            self.image_dir is not None
            and self.dataset is not None
            and self.dataset != DEFAULT_DATASET
        ):
            raise ValueError("--image-dir and --dataset are mutually exclusive")
        if self.image_dir is not None:
            self.dataset = None
        return self


def run_once(
    model,
    processor,
    images: list[Image.Image],
    dev: str,
    max_new_tokens: int | None,
) -> tuple[float, float, float, float, int, float, float]:
    """One encode+decode pass.

    Returns (preprocess_ms, encode_ms, decode_ms, e2e_ms, n_tokens,
             encode_peak_mb, decode_peak_mb).
    """
    t_e2e = CudaTimer()
    t_e2e.start()

    t0 = time.perf_counter()
    pixel_values = (
        processor(images, return_tensors="pt").pixel_values.to(dev).to(model.dtype)
    )
    decoder_input_ids = processor.tokenizer(
        TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(dev)
    decoder_input_ids = decoder_input_ids.expand(len(images), -1)
    preprocess_ms = (time.perf_counter() - t0) * 1000.0

    t = CudaTimer()

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t.start()
    with torch.no_grad():
        enc_out = model.encoder(pixel_values, return_dict=True)
    encode_ms = t.stop()
    encode_peak_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if dev == "cuda" else 0.0
    )

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gen_kwargs: dict = dict(
        pixel_values=pixel_values,
        decoder_input_ids=decoder_input_ids,
        encoder_outputs=enc_out,
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    if max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = max_new_tokens
        del gen_kwargs["max_length"]
    t.start()
    with torch.no_grad():
        seqs = model.generate(**gen_kwargs).sequences
    decode_ms = t.stop()
    decode_peak_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if dev == "cuda" else 0.0
    )

    e2e_ms = t_e2e.stop()

    prompt_len = decoder_input_ids.shape[1]
    n_tokens = 0
    for row in seqs[:, prompt_len:]:
        eos = (row == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        n_tokens += int(eos[0].item()) + 1 if len(eos) > 0 else len(row)

    return (
        preprocess_ms,
        encode_ms,
        decode_ms,
        e2e_ms,
        n_tokens,
        encode_peak_mb,
        decode_peak_mb,
    )


def build_results(cfg: Settings, pool, lists: dict) -> dict:

    pre = lists["preprocess_ms"]
    enc = lists["encode_ms"]
    dec = lists["decode_ms"]
    e2e = lists["e2e_ms"]
    tok = lists["n_tokens"]
    dec_per_tok = lists["decode_ms_per_token"]

    mean_e2e_s = statistics.mean(e2e) / 1000.0
    mean_tok = statistics.mean(tok)

    return {
        "schema_version": 2,
        "script": "bench_dataset.py",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "args": {
            "model": cfg.model,
            "device": cfg.device,
            "warmup": cfg.warmup,
            "runs": cfg.runs,
            "pool": cfg.pool,
            "batch_size": cfg.batch_size,
            "max_new_tokens": cfg.max_new_tokens,
            "dataset": cfg.dataset,
            "dataset_split": cfg.dataset_split,
            "image_dir": cfg.image_dir,
            "no_patch": cfg.no_patch,
        },
        "system": system_info(),
        "pool": {
            "size": len(pool),
        },
        "runs": lists,
        "summary": {
            "preprocess_ms": stat(pre),
            "encode_ms": stat(enc),
            "decode_ms": stat(dec),
            "decode_ms_per_token": stat(dec_per_tok),
            "e2e_ms": stat(e2e),
            "n_tokens": stat([float(t) for t in tok]),
            "encode_peak_gpu_mb": round(max(lists["encode_peak_mb"]), 1),
            "decode_peak_gpu_mb": round(max(lists["decode_peak_mb"]), 1),
            "samples_per_sec": round(cfg.batch_size / mean_e2e_s, 3),
            "tokens_per_sec": round(mean_tok / (statistics.mean(dec) / 1000.0), 2),
        },
    }


def main():
    cfg = Settings()
    dev = cfg.device

    print(f"Loading model ({cfg.model})...")
    model, processor = load_model(cfg.model, dev)
    apply_patch(model, dev, cfg.no_patch)

    print("Loading image pool...")
    pool = load_pool(
        cfg.pool, cfg.dataset, cfg.dataset_split, cfg.image_column, cfg.image_dir
    )

    print(f"Warmup ({cfg.warmup} run(s))...")
    for _ in range(cfg.warmup):
        run_once(
            model,
            processor,
            sample_batch(pool, cfg.batch_size),
            dev,
            cfg.max_new_tokens,
        )

    print(f"Measuring ({cfg.runs} run(s), batch={cfg.batch_size})...")
    pre_list: list[float] = []
    enc_list: list[float] = []
    dec_list: list[float] = []
    e2e_list: list[float] = []
    tok_list: list[int] = []
    dec_per_tok_list: list[float] = []
    enc_peak_list: list[float] = []
    dec_peak_list: list[float] = []

    for _ in range(cfg.runs):
        imgs = sample_batch(pool, cfg.batch_size)
        pre, enc, dec, e2e, n_tok, enc_peak, dec_peak = run_once(
            model, processor, imgs, dev, cfg.max_new_tokens
        )
        pre_list.append(pre)
        enc_list.append(enc)
        dec_list.append(dec)
        e2e_list.append(e2e)
        tok_list.append(n_tok)
        dec_per_tok_list.append(dec / n_tok if n_tok > 0 else 0.0)
        enc_peak_list.append(enc_peak)
        dec_peak_list.append(dec_peak)

        if cfg.save:
            atomic_save_json(
                cfg.save,
                build_results(
                    cfg,
                    pool,
                    {
                        "preprocess_ms": pre_list,
                        "encode_ms": enc_list,
                        "decode_ms": dec_list,
                        "decode_ms_per_token": dec_per_tok_list,
                        "e2e_ms": e2e_list,
                        "n_tokens": tok_list,
                        "encode_peak_mb": enc_peak_list,
                        "decode_peak_mb": dec_peak_list,
                    },
                ),
            )

    def _mean_std(vals: list[float], unit: str = "ms") -> str:
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{m:7.1f} ± {s:5.1f} {unit}"

    w = 12
    tok_mean = statistics.mean(tok_list)
    tok_std = statistics.stdev(tok_list) if len(tok_list) > 1 else 0.0
    tok_per_sec = tok_mean / (statistics.mean(dec_list) / 1000.0)

    print()
    print("=" * 55)
    print(f"  {'device':<{w}} : {dev}  |  batch={cfg.batch_size}  |  N={cfg.runs}")
    if torch.cuda.is_available():
        print(f"  {'GPU':<{w}} : {torch.cuda.get_device_name(0)}")
    print("=" * 55)
    enc_peak_str = (
        f"  [peak {statistics.mean(enc_peak_list):.0f} MB]" if dev == "cuda" else ""
    )
    dec_peak_str = (
        f"  [peak {statistics.mean(dec_peak_list):.0f} MB]" if dev == "cuda" else ""
    )
    print(f"  {'preprocess':<{w}} : {_mean_std(pre_list)}")
    print(f"  {'encode':<{w}} : {_mean_std(enc_list)}{enc_peak_str}")
    print(f"  {'decode':<{w}} : {_mean_std(dec_per_tok_list, 'ms/tok')}{dec_peak_str}")
    print(
        f"  {'tokens':<{w}} : {tok_mean:.0f} ± {tok_std:.0f}  (range {min(tok_list)}–{max(tok_list)})"
    )
    print(
        f"  {'throughput':<{w}} : {tok_per_sec:.0f} tok/s  |  {cfg.batch_size / (statistics.mean(e2e_list) / 1000.0):.2f} samples/s"
    )
    print("=" * 55)


if __name__ == "__main__":
    main()
