from __future__ import annotations

import logging
from datetime import datetime, timezone

import torch
from PIL import Image

from .config import BenchmarkConfig
from .memory import capture_memory_snapshot, reset_memory_stats
from .metrics import AggregatedResult, BatchSweepResult, RunResult, compute_stats
from .model import ModelBundle
from .profiler import FLOPProfile
from .timer import make_timer

logger = logging.getLogger("benchmark.runner")


def _select_images(
    image_pool: list[Image.Image],
    pool_indices: list[int],
    run_k: int,
    batch_size: int,
) -> tuple[list[Image.Image], list[int]]:
    """Round-robin selection from the image pool for run k."""
    n = len(image_pool)
    positions = [(run_k * batch_size + i) % n for i in range(batch_size)]
    return [image_pool[p] for p in positions], [pool_indices[p] for p in positions]


class BenchmarkRunner:
    def __init__(
        self,
        config: BenchmarkConfig,
        bundle: ModelBundle,
        flop_profile: FLOPProfile | None = None,
    ) -> None:
        self.config = config
        self.bundle = bundle
        self.flop_profile = flop_profile

    def run_single(
        self,
        images: list[Image.Image],
        image_indices: list[int],
        run_index: int,
    ) -> RunResult | None:
        """Execute one full inference pass over a batch and collect all metrics.

        Returns None on OOM; caller increments oom_count and recovers.
        """
        cfg = self.config
        dev = cfg.device
        B = len(images)

        try:
            reset_memory_stats(dev)

            t_e2e = make_timer(dev)
            t_pre = make_timer(dev)
            t_enc = make_timer(dev)
            t_dec = make_timer(dev)
            t_post = make_timer(dev)

            t_e2e.start()

            t_pre.start()
            pre = self.bundle.preprocess(images)
            pixel_values = pre.pixel_values
            decoder_input_ids = pre.decoder_input_ids
            t_pre.stop()

            t_enc.start()
            encoder_outputs = self.bundle.encode(pixel_values)
            t_enc.stop()

            t_dec.start()
            sequences = self.bundle.decode(
                pixel_values, encoder_outputs, decoder_input_ids
            )
            t_dec.stop()

            t_post.start()
            postprocess_results = self.bundle.postprocess(sequences)
            t_post.stop()

            t_e2e.stop()

            if dev == "cuda":
                torch.cuda.synchronize()

            mem = capture_memory_snapshot(dev)

            preprocess_ms = t_pre.elapsed_ms()
            encoder_ms = t_enc.elapsed_ms()
            decoder_ms = t_dec.elapsed_ms()
            postprocess_ms = t_post.elapsed_ms()
            end_to_end_ms = t_e2e.elapsed_ms()

            prompt_len = decoder_input_ids.shape[-1]
            actual_lens, max_len, sum_len, dec_efficiency = self.bundle.count_tokens(
                sequences, prompt_len
            )

            tokens_per_second = sum_len / (decoder_ms / 1000.0)
            docs_per_second = B * 1000.0 / end_to_end_ms
            per_sample_latency_ms = end_to_end_ms / B
            orig_megapixels_per_second = (
                pre.orig_megapixels_mean * B * 1000.0 / end_to_end_ms
            )
            processed_megapixels_per_second = (
                pre.processed_megapixels * B * 1000.0 / end_to_end_ms
            )

            # FLOPs (zero when no profile is available, e.g. CPU runs)
            fp = self.flop_profile
            if fp is not None:
                enc_flops = B * fp.encoder_flops_per_image
                dec_flops = max_len * B * fp.decoder_flops_per_step_per_sample
                total_flops = enc_flops + dec_flops
                encoder_tflops = enc_flops / (encoder_ms / 1000.0) / 1e12
                decoder_tflops = dec_flops / (decoder_ms / 1000.0) / 1e12
            else:
                total_flops = 0
                encoder_tflops = 0.0
                decoder_tflops = 0.0

            is_valid_json = all(parsed is not None for _, parsed in postprocess_results)
            raw_output = " | ".join(text for text, _ in postprocess_results)

            result = RunResult(
                run_index=run_index,
                batch_size=B,
                image_indices=image_indices,
                preprocess_ms=preprocess_ms,
                encoder_ms=encoder_ms,
                decoder_ms=decoder_ms,
                postprocess_ms=postprocess_ms,
                end_to_end_ms=end_to_end_ms,
                per_sample_latency_ms=per_sample_latency_ms,
                docs_per_second=docs_per_second,
                num_generated_tokens=sum_len,
                max_generated_tokens_in_batch=max_len,
                tokens_per_second=tokens_per_second,
                decoder_efficiency=dec_efficiency,
                orig_image_width_mean=pre.orig_image_width_mean,
                orig_image_height_mean=pre.orig_image_height_mean,
                orig_megapixels_mean=pre.orig_megapixels_mean,
                processed_image_width=pre.processed_image_width,
                processed_image_height=pre.processed_image_height,
                processed_megapixels=pre.processed_megapixels,
                orig_megapixels_per_second=orig_megapixels_per_second,
                processed_megapixels_per_second=processed_megapixels_per_second,
                peak_gpu_memory_mb=mem.peak_allocated_mb,
                allocated_gpu_memory_mb=mem.allocated_mb,
                reserved_gpu_memory_mb=mem.reserved_mb,
                total_flops=total_flops,
                encoder_tflops=encoder_tflops,
                decoder_tflops=decoder_tflops,
                is_valid_json=is_valid_json,
                raw_output=raw_output,
            )

            logger.info(
                "Run %d/%d | B=%d imgs=%s | e2e=%.1fms lat/doc=%.1fms "
                "enc=%.1fms dec=%.1fms | tokens=%d tok/s=%.1f dec_eff=%.3f "
                "| peak_mem=%.1fMB | enc_TFLOPS=%.3f dec_TFLOPS=%.3f | valid_json=%s",
                run_index + 1,
                cfg.num_runs,
                B,
                image_indices,
                end_to_end_ms,
                per_sample_latency_ms,
                encoder_ms,
                decoder_ms,
                sum_len,
                tokens_per_second,
                dec_efficiency,
                mem.peak_allocated_mb,
                encoder_tflops,
                decoder_tflops,
                is_valid_json,
            )
            return result

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error(
                "Run %d (B=%d): CUDA out of memory — skipped.", run_index + 1, B
            )
            return None

    def run(
        self,
        image_pool: list[Image.Image],
        pool_indices: list[int],
        run_id: str,
    ) -> AggregatedResult:
        """Warmup + measurement loop using the image pool."""
        cfg = self.config
        B = cfg.batch_size

        baseline_mb = capture_memory_snapshot(cfg.device).allocated_mb

        logger.info(
            "Starting benchmark: model=%s device=%s batch_size=%d "
            "pool=%d images warmup=%d runs=%d",
            cfg.model_id,
            cfg.device,
            B,
            len(image_pool),
            cfg.warmup_runs,
            cfg.num_runs,
        )

        for i in range(cfg.warmup_runs):
            imgs, idxs = _select_images(image_pool, pool_indices, i, B)
            self.run_single(imgs, idxs, run_index=-(cfg.warmup_runs - i))
            logger.info("Warmup %d/%d complete.", i + 1, cfg.warmup_runs)

        results: list[RunResult] = []
        oom_count = 0

        for i in range(cfg.num_runs):
            imgs, idxs = _select_images(image_pool, pool_indices, i, B)
            result = self.run_single(imgs, idxs, run_index=i)
            if result is None:
                oom_count += 1
            else:
                results.append(result)

        if not results:
            raise RuntimeError(
                f"All {cfg.num_runs} measurement runs failed (OOM). "
                "Try a smaller batch size or --device cpu."
            )

        logger.info(
            "Benchmark complete (B=%d): %d/%d runs successful, %d OOM.",
            B,
            len(results),
            cfg.num_runs,
            oom_count,
        )

        return compute_stats(
            config=cfg,
            runs=results,
            oom_count=oom_count,
            baseline_gpu_allocated_mb=baseline_mb,
            run_id=run_id,
        )

    def run_sweep(
        self,
        image_pool: list[Image.Image],
        pool_indices: list[int],
        run_id: str,
    ) -> BatchSweepResult:
        """Iterate over config.batch_sizes and run a full benchmark for each."""
        from .metrics import _system_info

        cfg = self.config
        original_batch_size = cfg.batch_size
        results_by_batch_size: dict[int, AggregatedResult | None] = {}

        for B in cfg.batch_sizes:
            logger.info("=== Sweep: batch_size=%d ===", B)
            cfg.batch_size = B
            sub_run_id = f"{run_id}_bs{B}"
            try:
                agg = self.run(image_pool, pool_indices, sub_run_id)
                results_by_batch_size[B] = agg
            except RuntimeError as exc:
                logger.error("Sweep B=%d: all runs failed — %s", B, exc)
                results_by_batch_size[B] = None

        cfg.batch_size = original_batch_size

        return BatchSweepResult(
            run_id=run_id,
            config=cfg,
            system_info=_system_info(),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            results_by_batch_size=results_by_batch_size,
        )
