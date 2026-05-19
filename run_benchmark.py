"""CLI entrypoint for the Donut inference benchmark."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import torch
from datasets import load_dataset

from benchmark import (
    BenchmarkRunner,
    ModelBundle,
    build_arg_parser,
    config_from_args,
    profile_flops,
)
from benchmark.output import (
    print_summary,
    print_sweep_summary,
    setup_logging,
    write_csv,
    write_json,
    write_sweep_csv,
    write_sweep_json,
)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)

    run_id = "benchmark_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    os.makedirs(config.output_dir, exist_ok=True)
    logger = setup_logging(config.output_dir, run_id)

    logger.info("Run ID: %s", run_id)
    logger.info("Config: %s", config)

    if config.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU.")
        config.device = "cpu"

    logger.info(
        "Loading dataset %s[%s] pool_size=%s …",
        config.dataset_name,
        config.dataset_split,
        config.pool_size or "all",
    )
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    n = len(dataset) if config.pool_size == 0 else min(config.pool_size, len(dataset))
    pool_indices = list(range(n))
    image_pool = [dataset[i]["image"] for i in pool_indices]
    logger.info("Loaded %d images into pool.", len(image_pool))

    logger.info("Loading model %s …", config.model_id)
    bundle = ModelBundle.load(
        model_id=config.model_id,
        device=config.device,
        task_prompt=config.task_prompt,
    )

    # FLOPs calibration (skipped on CPU — profiler overhead not worthwhile there)
    flop_profile = None
    if config.device == "cuda":
        logger.info("Running FLOPs calibration pass …")
        flop_profile = profile_flops(bundle, image_pool[0])

    runner = BenchmarkRunner(config=config, bundle=bundle, flop_profile=flop_profile)

    if config.batch_sizes:
        # ---- sweep mode ----
        sweep = runner.run_sweep(image_pool, pool_indices, run_id)

        sweep_json = os.path.join(config.output_dir, f"{run_id}_sweep.json")
        sweep_csv = os.path.join(config.output_dir, f"{run_id}_sweep.csv")
        write_sweep_json(sweep, sweep_json)
        write_sweep_csv(sweep, sweep_csv)
        print_sweep_summary(sweep)

        # Also write per-batch-size per-run CSVs for detailed analysis
        for B, agg in sweep.results_by_batch_size.items():
            if agg is not None:
                sub_csv = os.path.join(config.output_dir, f"{run_id}_bs{B}_runs.csv")
                write_csv(agg.runs, sub_csv)

    else:
        # ---- single batch size mode ----
        result = runner.run(image_pool, pool_indices, run_id)

        json_path = os.path.join(config.output_dir, f"{run_id}.json")
        csv_path = os.path.join(config.output_dir, f"{run_id}.csv")
        write_json(result, json_path)
        write_csv(result.runs, csv_path)
        print_summary(result)

    logger.info("All outputs written to %s/", config.output_dir)


if __name__ == "__main__":
    main()
