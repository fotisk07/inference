from __future__ import annotations

import argparse
from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    model_id: str = "naver-clova-ix/donut-base-finetuned-cord-v2"
    task_prompt: str = "<s_cord-v2>"
    device: str = "cuda"
    # Active batch size for the current run. In sweep mode this is overwritten
    # for each point in batch_sizes.
    batch_size: int = 1
    # If non-empty, run_benchmark.py enters sweep mode and iterates over these.
    batch_sizes: list[int] = field(default_factory=list)
    num_runs: int = 50
    warmup_runs: int = 5
    output_dir: str = "logs"
    dataset_name: str = "naver-clova-ix/cord-v2"
    dataset_split: str = "test"
    # Number of images to load from the split into the pool (0 = whole split).
    pool_size: int = 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rigorous Donut inference benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-id", default=BenchmarkConfig.model_id)
    p.add_argument("--task-prompt", default=BenchmarkConfig.task_prompt)
    p.add_argument("--device", default=BenchmarkConfig.device, choices=["cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=BenchmarkConfig.batch_size)
    p.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[],
        help="If given, run a sweep over these batch sizes (enables sweep mode).",
    )
    p.add_argument("--num-runs", type=int, default=BenchmarkConfig.num_runs)
    p.add_argument("--warmup-runs", type=int, default=BenchmarkConfig.warmup_runs)
    p.add_argument("--output-dir", default=BenchmarkConfig.output_dir)
    p.add_argument("--dataset-name", default=BenchmarkConfig.dataset_name)
    p.add_argument("--dataset-split", default=BenchmarkConfig.dataset_split)
    p.add_argument(
        "--pool-size",
        type=int,
        default=BenchmarkConfig.pool_size,
        help="Number of images to load from the split (0 = whole split).",
    )
    return p


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        model_id=args.model_id,
        task_prompt=args.task_prompt,
        device=args.device,
        batch_size=args.batch_size,
        batch_sizes=args.batch_sizes,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        pool_size=args.pool_size,
    )
