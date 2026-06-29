"""Kernel-level profile of one generate() and one train step (SKELETON).

Purpose (research/decode-profiler): the wall-clock bench (scripts/inference/
bench_speed.py) tells us decode is slow but not WHY. This wraps a single
generate() and a single train step in torch.profiler so we can see, per CUDA
kernel: how much is kernel compute vs how much is launch gap / CPU-bound
dispatch. The decision rule for the compile branches lives in
donut/research/decode-profiler.md.

This is a runnable skeleton: it emits a chrome trace + a key_averages table.
The TODO markers are where the actual analysis (launch-gap extraction, roofline)
gets fleshed out tomorrow.

Run:
    uv run donut/scripts/profiling/profile_decode.py --backend sdpa
    # then open the .json trace in chrome://tracing or Perfetto
"""

from pathlib import Path
from typing import Literal

import torch
import typer
from torch.profiler import ProfilerActivity, profile

from donut.accel import apply_accel, check_accel, revert_accel
from donut.constants import (
    DEFAULT_IMAGE_SIZE_STR,
    DEFAULT_MAX_NEW_TOKENS,
    GLOBAL_OUT_DIR,
    MODEL_ID,
)
from donut.model import (
    autocast,
    decoder_start_ids,
    decoder_vocab_size,
    init_shift_tokens_from_decoder,
    load_baseline_model,
    pad_token_id,
    set_encoder_image_size,
)
from donut.runio import parse_image_sizes, resolve_device_dtype
from donut.synthetic import make_pixel_values

app = typer.Typer(add_completion=False)


def _activities() -> list[ProfilerActivity]:
    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    return acts


def _summarize(prof, sort_key: str) -> None:
    """Print the top kernels and TODO: extract launch-gap / bandwidth signal.

    key_averages() gives per-op CUDA time and CPU time. The gap between summed
    CUDA time and wall time is the launch-overhead we care about at query_len=1.
    """
    print(prof.key_averages().table(sort_by=sort_key, row_limit=25))
    # TODO(tomorrow): compute total_cuda_time vs wall_time -> launch-gap fraction.
    # TODO(tomorrow): bucket kernels by name (gemm vs elementwise vs memcpy) to
    #                 separate compute-bound from bandwidth-bound time.
    # TODO(tomorrow): roofline — map decoder gemm shapes (q=1, kv=t) to achieved
    #                 GB/s and compare against device peak to confirm bw-bound.


@app.command()
def main(
    model_id: str = MODEL_ID,
    device: str | None = None,
    dtype: Literal["bf16", "f16", "f32"] = "bf16",
    backend: str = "sdpa",
    image_size: str = DEFAULT_IMAGE_SIZE_STR,
    batch_size: int = 1,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = 42,
    out: Path = GLOBAL_OUT_DIR / "results" / "profile_decode",
    skip_train: bool = False,
) -> None:
    """Profile one generate() (decode) and one train step for a single backend."""
    device, torch_dtype = resolve_device_dtype(device, dtype)
    (h, w) = parse_image_sizes(image_size)[0]
    out.mkdir(parents=True, exist_ok=True)

    model, model_id = load_baseline_model(model_id, device, torch_dtype)
    set_encoder_image_size(model, h, w)
    init_shift_tokens_from_decoder(model)
    apply_accel(model, backend)
    check_accel(model, backend)

    pixel_values = make_pixel_values(model, batch_size=batch_size, seed=seed)
    decoder_input_ids = decoder_start_ids(model, batch_size=batch_size)
    pad_id = pad_token_id(model)

    # ── Inference (decode) profile ─────────────────────────────────────────────
    # Warm up once so we capture steady-state kernels, not first-call autotuning.
    model.eval()
    with torch.no_grad():
        model.generate(  # ty: ignore
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            use_cache=True,
        )

    with profile(activities=_activities(), record_shapes=True) as prof:
        with torch.no_grad():
            model.generate(  # ty: ignore
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                use_cache=True,
            )
    trace = out / f"decode__{backend}__{h}x{w}__bs{batch_size}.json"
    prof.export_chrome_trace(str(trace))
    print(f"\n=== DECODE profile ({backend}) -> {trace} ===")
    _summarize(
        prof, "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    )

    if skip_train:
        revert_accel(model)
        return

    # ── Training step profile ──────────────────────────────────────────────────
    model.train()
    vocab = decoder_vocab_size(model)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    gen = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    labels = torch.randint(
        0,
        vocab,
        (batch_size, max_new_tokens),
        device=next(model.parameters()).device,
        generator=gen,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-9)

    def train_step():
        optimizer.zero_grad()
        with autocast(device_type, "bf16" if dtype == "bf16" else "f32"):
            loss = model(pixel_values=pixel_values, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    train_step()  # warmup
    with profile(activities=_activities(), record_shapes=True) as prof:
        train_step()
    trace = out / f"train__{backend}__{h}x{w}__bs{batch_size}.json"
    prof.export_chrome_trace(str(trace))
    print(f"\n=== TRAIN profile ({backend}) -> {trace} ===")
    _summarize(
        prof, "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    )

    revert_accel(model)


if __name__ == "__main__":
    app()
