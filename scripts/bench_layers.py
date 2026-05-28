"""Donut layer-level profiler — low-level component timing over a dataset.

Times each encoder stage (Swin) and each decoder layer (MBart) using CUDA events,
aggregated across N images from a real dataset.

Encoder metrics: absolute ms per stage (independent of sequence length).
Decoder metrics: ms per token per layer (normalized, since the decoder is
autoregressive and each hook fires once per generated token).

Prints a compact two-table summary; saves full per-image raw data to JSON.

Requires CUDA. Use bench.py / diagnose.py for CPU profiling.

Usage:
    uv run scripts/bench_layers.py [--n-images 10] [--pool 20] [--save layers.json]
    uv run scripts/bench_layers.py --dataset naver-clova-ix/cord-v2 --n-images 20
    uv run scripts/bench_layers.py --image-dir /path/to/images --n-images 5
    uv run scripts/bench_layers.py --no-patch --save layers.json
"""

import datetime
import sys

import torch
from PIL import Image
from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict
from tqdm import tqdm

from inference.constants import DEFAULT_DATASET, TASK_PROMPT
from inference.data import load_pool, sample_batch
from inference.model import apply_patch, load_model
from inference.saving import atomic_save_json
from inference.settings import BenchSettings
from inference.stats import stat, system_info
from inference.timing import LayerTimer


class Settings(BenchSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True, env_prefix="BENCH_", cli_prog_name="bench_layers"
    )
    n_images: int = Field(default=10, description="Number of images to profile")
    pool: int = Field(default=20, description="Size of image pool to sample from")
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


def register_timers(model) -> tuple[list[LayerTimer], list]:
    """Attach LayerTimers to encoder stages and decoder layers."""
    timers: list[LayerTimer] = []
    handles: list = []

    def attach(module, name: str) -> LayerTimer:
        t = LayerTimer(name)
        handles.append(module.register_forward_pre_hook(t.pre_hook))
        handles.append(module.register_forward_hook(t.post_hook))
        timers.append(t)
        return t

    attach(model.encoder.embeddings, "enc:patch_embed")
    for i, stage in enumerate(model.encoder.encoder.layers):
        attach(stage, f"enc:stage{i}")

    attach(model.decoder.model.decoder.embed_tokens, "dec:embed_tokens")
    for i, layer in enumerate(model.decoder.model.decoder.layers):
        attach(layer, f"dec:layer{i}")
    attach(model.decoder.lm_head, "dec:lm_head")

    return timers, handles


def remove_handles(handles: list) -> None:
    for h in handles:
        h.remove()


def run_pass(model, processor, pixel_values, decoder_input_ids):
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


def profile_one_image(
    model, processor, image: Image.Image, dev: str, timers: list[LayerTimer]
) -> dict:
    """Run one profiled pass and return per-layer timing for this image."""
    for t in timers:
        t.reset()

    pixel_values = (
        processor(image, return_tensors="pt").pixel_values.to(dev).to(model.dtype)
    )
    decoder_input_ids = processor.tokenizer(
        TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(dev)

    _, seqs = run_pass(model, processor, pixel_values, decoder_input_ids)

    torch.cuda.synchronize()
    by_name: dict[str, list[float]] = {t.name: t.elapsed_no_sync() for t in timers}

    prompt_len = decoder_input_ids.shape[1]
    row = seqs[0, prompt_len:]
    eos = (row == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    n_tokens = int(eos[0].item()) + 1 if len(eos) > 0 else len(row)

    enc_keys = [k for k in by_name if k.startswith("enc:")]
    encoder: dict = {}
    for key in enc_keys:
        field = key.removeprefix("enc:")
        encoder[field] = round(by_name[key][0], 3)
    encoder["total"] = round(sum(encoder[v] for v in encoder), 3)

    dec_keys = [k for k in by_name if k.startswith("dec:")]
    decoder: dict = {"layers": {}}
    for key in dec_keys:
        samples = by_name[key]
        total_ms = sum(samples)
        label = key.removeprefix("dec:")
        decoder["layers"][label] = {
            "ms_per_token": round(total_ms / n_tokens if n_tokens > 0 else 0.0, 4),
            "total_ms": round(total_ms, 3),
            "samples": [round(v, 4) for v in samples],
        }
    dec_total_ms = sum(v["total_ms"] for v in decoder["layers"].values())
    decoder["total_ms_per_token"] = round(
        dec_total_ms / n_tokens if n_tokens > 0 else 0.0, 4
    )

    return {"n_tokens": n_tokens, "encoder": encoder, "decoder": decoder}


def aggregate(per_image: list[dict]) -> dict:
    """Compute stats per layer across all images."""
    enc_keys = [k for k in per_image[0]["encoder"] if k != "total"]
    dec_keys = list(per_image[0]["decoder"]["layers"].keys())

    enc_summary: dict = {}
    enc_totals = [img["encoder"]["total"] for img in per_image]
    total_mean = stat(enc_totals)["mean"]
    for key in enc_keys:
        vals = [img["encoder"][key] for img in per_image]
        d = stat(vals)
        d["pct"] = round(100.0 * d["mean"] / total_mean, 1) if total_mean > 0 else 0.0
        enc_summary[key] = d
    enc_summary["total"] = stat(enc_totals)

    dec_summary: dict = {}
    dec_totals = [img["decoder"]["total_ms_per_token"] for img in per_image]
    total_mpt_mean = stat(dec_totals)["mean"]
    for key in dec_keys:
        vals = [img["decoder"]["layers"][key]["ms_per_token"] for img in per_image]
        d = stat(vals)
        d["pct"] = (
            round(100.0 * d["mean"] / total_mpt_mean, 1) if total_mpt_mean > 0 else 0.0
        )
        dec_summary[key] = d
    dec_summary["total"] = stat(dec_totals)

    n_tokens_vals = [float(img["n_tokens"]) for img in per_image]
    return {
        "n_images": len(per_image),
        "n_tokens": stat(n_tokens_vals),
        "encoder": enc_summary,
        "decoder": dec_summary,
    }


def print_summary(summary: dict) -> None:
    enc = summary["encoder"]
    dec = summary["decoder"]
    n = summary["n_images"]
    tok = summary["n_tokens"]
    enc_total = enc["total"]["mean"]
    dec_total = dec["total"]["mean"]
    col = 14

    print()
    print("=" * 55)
    print(f"  Encoder  (mean {enc_total:.1f} ms  over {n} images)")
    print("  " + "-" * 51)
    enc_keys = [k for k in enc if k != "total"]
    for key in sorted(enc_keys, key=lambda k: enc[k]["mean"], reverse=True):
        d = enc[key]
        print(f"  {key:<{col}} {d['mean']:7.1f} ms   {d['pct']:5.1f}%  ±{d['std']:.1f}")
    print()
    print(
        f"  Decoder  (mean {dec_total:.2f} ms/tok  |  {tok['mean']:.0f} ± {tok['std']:.0f} tokens)"
    )
    print("  " + "-" * 51)
    dec_keys = [k for k in dec if k != "total"]
    for key in sorted(dec_keys, key=lambda k: dec[k]["mean"], reverse=True):
        d = dec[key]
        print(
            f"  {key:<{col}} {d['mean']:6.3f} ms/tok  {d['pct']:5.1f}%  ±{d['std']:.3f}"
        )
    print("=" * 55)


def build_results(cfg: Settings, input_shape, per_image, summary) -> dict:
    return {
        "schema_version": 2,
        "script": "bench_layers.py",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "args": {
            "model": cfg.model,
            "device": cfg.device,
            "warmup": cfg.warmup,
            "n_images": cfg.n_images,
            "pool": cfg.pool,
            "dataset": cfg.dataset,
            "dataset_split": cfg.dataset_split,
            "image_dir": cfg.image_dir,
            "no_patch": cfg.no_patch,
        },
        "system": system_info(),
        "input_shape": input_shape,
        "per_image": per_image,
        "summary": summary,
    }


def main():
    cfg = Settings()
    dev = cfg.device
    if dev != "cuda":
        sys.exit(
            "bench_layers.py requires CUDA (CUDA events needed for per-layer timing)"
        )

    print(f"  GPU : {torch.cuda.get_device_name(0)}")

    print(f"Loading model ({cfg.model})...")
    model, processor = load_model(cfg.model, dev)
    apply_patch(model, dev, cfg.no_patch)

    print("Loading image pool...")
    pool = load_pool(
        cfg.pool, cfg.dataset, cfg.dataset_split, cfg.image_column, cfg.image_dir
    )

    print(f"Warmup ({cfg.warmup} run(s))...")
    for _ in range(cfg.warmup):
        [img] = sample_batch(pool, 1)
        pv = processor(img, return_tensors="pt").pixel_values.to(dev).to(model.dtype)
        did = processor.tokenizer(
            TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(dev)
        run_pass(model, processor, pv, did)
    torch.cuda.synchronize()

    [sample_img] = sample_batch(pool, 1)
    input_shape = list(
        processor(sample_img, return_tensors="pt").pixel_values.shape
    )

    timers, handles = register_timers(model)

    per_image: list[dict] = []
    for _ in tqdm(range(cfg.n_images), desc="profiling"):
        [img] = sample_batch(pool, 1)
        result = profile_one_image(model, processor, img, dev, timers)
        per_image.append(result)

        if cfg.save:
            atomic_save_json(
                cfg.save,
                build_results(cfg, input_shape, per_image, aggregate(per_image)),
            )

    remove_handles(handles)

    summary = aggregate(per_image)
    print_summary(summary)


if __name__ == "__main__":
    main()
