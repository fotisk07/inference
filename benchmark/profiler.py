from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from PIL import Image

from .model import ModelBundle

logger = logging.getLogger("benchmark.profiler")

_PROFILER_ACTIVITIES = [
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
]


@dataclass
class FLOPProfile:
    """FLOPs per logical unit of work, calibrated from a single profiling pass.

    These are lower bounds: torch.profiler counts standard PyTorch ops but may
    miss fused or custom CUDA kernels. For transformers==4.37.2 with Donut
    (no Flash Attention), coverage is high enough to be informative.
    """

    encoder_flops_per_image: int
    # Average FLOPs per decoder step per sample in the batch. Self-attention
    # FLOPs grow with sequence position, so this is an average over all steps.
    decoder_flops_per_step_per_sample: int


def _sum_flops(prof: torch.profiler.profile) -> int:
    return int(sum(e.flops for e in prof.key_averages() if e.flops > 0))


def profile_flops(bundle: ModelBundle, image: Image.Image) -> FLOPProfile:
    """Run two separate one-shot profiling passes and extract FLOPs.

    Pass 1: encoder only  →  encoder_flops_per_image
    Pass 2: full generate →  decoder FLOPs, divided by steps taken

    This function is called once before the benchmark loop; it adds no overhead
    to the actual benchmark runs.
    """
    # ---- warm up CUDA so profiler times are stable ----
    pre = bundle.preprocess([image])
    pixel_values, decoder_input_ids = pre.pixel_values, pre.decoder_input_ids
    with torch.no_grad():
        enc_out = bundle.encode(pixel_values)
        _ = bundle.decode(pixel_values, enc_out, decoder_input_ids)

    # ---- Pass 1: encoder FLOPs ----
    with torch.profiler.profile(
        activities=_PROFILER_ACTIVITIES,
        with_flops=True,
    ) as enc_prof:
        pre = bundle.preprocess([image])
        pixel_values, decoder_input_ids = pre.pixel_values, pre.decoder_input_ids
        with torch.no_grad():
            enc_out = bundle.encode(pixel_values)

    encoder_flops = _sum_flops(enc_prof)
    logger.info(
        "Profiler: encoder FLOPs per image = %d (%.3f GFLOPS)",
        encoder_flops,
        encoder_flops / 1e9,
    )

    # ---- Pass 2: full generate FLOPs (encoder already provided) ----
    # Re-use enc_out from the profiled encoder pass — we profile only the
    # decoder (generate call) here to isolate its FLOPs.
    pre = bundle.preprocess([image])
    pixel_values, decoder_input_ids = pre.pixel_values, pre.decoder_input_ids
    with torch.no_grad():
        enc_out_for_dec = bundle.encode(pixel_values)

    sequences_ref = None
    with torch.profiler.profile(
        activities=_PROFILER_ACTIVITIES,
        with_flops=True,
    ) as dec_prof:
        with torch.no_grad():
            seqs = bundle.decode(pixel_values, enc_out_for_dec, decoder_input_ids)
        sequences_ref = seqs

    decoder_total_flops = _sum_flops(dec_prof)

    # Count actual steps taken (tokens generated minus the prompt token).
    prompt_len = decoder_input_ids.shape[-1]
    _, max_len, _, _ = bundle.count_tokens(sequences_ref, prompt_len)
    num_steps = max(max_len, 1)

    decoder_flops_per_step = decoder_total_flops // num_steps

    logger.info(
        "Profiler: decoder total FLOPs = %d (%.3f GFLOPS) over %d steps → %d FLOPs/step",
        decoder_total_flops,
        decoder_total_flops / 1e9,
        num_steps,
        decoder_flops_per_step,
    )

    return FLOPProfile(
        encoder_flops_per_image=max(encoder_flops, 1),
        decoder_flops_per_step_per_sample=max(decoder_flops_per_step, 1),
    )
