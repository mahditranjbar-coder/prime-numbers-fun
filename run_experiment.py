from __future__ import annotations

import argparse
from pathlib import Path

from prime_lab import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Prime Pattern Lab pair-residual experiment.")
    parser.add_argument("--start", type=int, default=1_000_000)
    parser.add_argument("--limit", type=int, default=100_000_000)
    parser.add_argument("--max-gap", type=int, default=10_000)
    parser.add_argument("--block-ratio", type=float, default=1.5)
    parser.add_argument("--null-simulations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=104_729)
    parser.add_argument("--fft-workers", type=int, default=-1)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small range for a fast end-to-end smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.start = 100_000
        args.limit = 2_000_000
        args.max_gap = min(args.max_gap, 1_000)
        args.results_dir = str(Path(args.results_dir).with_name("quick_results"))
    config = ExperimentConfig(
        start=args.start,
        limit=args.limit,
        max_gap=args.max_gap,
        block_ratio=args.block_ratio,
        null_simulations=args.null_simulations,
        seed=args.seed,
        fft_workers=args.fft_workers,
        results_dir=args.results_dir,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()

