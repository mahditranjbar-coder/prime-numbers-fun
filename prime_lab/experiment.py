from __future__ import annotations

import gc
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import analyze_results
from .null_models import wheel_conditioned_sample
from .sieve import geometric_blocks, segmented_prime_mask, simple_sieve, von_mangoldt_segment
from .singular_series import even_offsets, pair_singular_series
from .statistics import autocorrelation_lags, inverse_log_square_integral


@dataclass(frozen=True)
class ExperimentConfig:
    start: int = 1_000_000
    limit: int = 100_000_000
    max_gap: int = 10_000
    block_ratio: float = 1.5
    wheel_primes: tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17, 19)
    null_simulations: int = 1
    seed: int = 104_729
    fft_workers: int = -1
    results_dir: str = "results"

    def validate(self) -> None:
        if not (100 <= self.start < self.limit):
            raise ValueError("require 100 <= start < limit")
        if self.max_gap < 2 or self.max_gap >= self.start:
            raise ValueError("require 2 <= max_gap < start")
        if self.block_ratio <= 1.0:
            raise ValueError("block_ratio must exceed 1")
        if self.null_simulations < 1:
            raise ValueError("null_simulations must be at least 1")
        if any(p < 2 for p in self.wheel_primes):
            raise ValueError("wheel primes must be positive primes")


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    config.validate()
    started = time.perf_counter()
    results_dir = Path(config.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    offsets = even_offsets(config.max_gap)
    singular = pair_singular_series(offsets)
    base_primes = simple_sieve(math.isqrt(config.limit - 1))
    blocks = list(geometric_blocks(config.start, config.limit, config.block_ratio))
    rng = np.random.default_rng(config.seed)

    (results_dir / "config.json").write_text(
        json.dumps({**asdict(config), "wheel_primes": list(config.wheel_primes)}, indent=2) + "\n"
    )

    pair_frames: list[pd.DataFrame] = []
    block_rows: list[dict[str, float | int]] = []
    normalized_gap_chunks: list[np.ndarray] = []
    previous_prime: int | None = None

    for block_id, (low, high) in enumerate(blocks):
        block_started = time.perf_counter()
        size = high - low
        print(f"[{block_id + 1:02d}/{len(blocks):02d}] sieve {low:,}..{high - 1:,}", flush=True)

        prime_mask = segmented_prime_mask(low, high, base_primes)
        prime_indices = np.flatnonzero(prime_mask)
        primes = low + prime_indices

        # Float64 keeps FFT roundoff comfortably below half a count even for
        # the largest configured blocks, so rounding recovers exact integers.
        count_corr = autocorrelation_lags(prime_mask.astype(np.float64), config.max_gap, workers=config.fft_workers)
        observed_count = np.rint(count_corr[offsets]).astype(np.int64)
        del count_corr

        mangoldt = von_mangoldt_segment(low, high, prime_mask, base_primes)
        weighted_corr = autocorrelation_lags(mangoldt, config.max_gap, workers=config.fft_workers)
        weighted_observed = weighted_corr[offsets].astype(np.float64, copy=True)
        del weighted_corr, mangoldt
        gc.collect()

        null_accumulator = np.zeros(len(offsets), dtype=np.float64)
        for _ in range(config.null_simulations):
            null_mask = wheel_conditioned_sample(low, high, config.wheel_primes, rng)
            null_corr = autocorrelation_lags(null_mask.astype(np.float64), config.max_gap, workers=config.fft_workers)
            null_accumulator += np.rint(null_corr[offsets])
            del null_corr, null_mask
        null_count = null_accumulator / config.null_simulations

        base_integral = inverse_log_square_integral(low, high)
        boundary_fraction = (size - offsets) / size
        predicted_count = singular * base_integral * boundary_fraction
        weighted_predicted = singular * (size - offsets)

        frame = pd.DataFrame(
            {
                "block_id": block_id,
                "low": low,
                "high": high,
                "midpoint": math.sqrt(low * high),
                "offset": offsets,
                "singular_series": singular,
                "observed_count": observed_count,
                "predicted_count": predicted_count,
                "count_ratio": observed_count / predicted_count,
                "count_residual": observed_count / predicted_count - 1.0,
                "weighted_observed": weighted_observed,
                "weighted_predicted": weighted_predicted,
                "psi_ratio": weighted_observed / weighted_predicted,
                "psi_residual": weighted_observed / weighted_predicted - 1.0,
                "null_count": null_count,
                "null_ratio": null_count / predicted_count,
                "null_residual": null_count / predicted_count - 1.0,
            }
        )
        pair_frames.append(frame)

        if len(primes):
            if previous_prime is not None:
                gap_primes = np.concatenate(([previous_prime], primes))
            else:
                gap_primes = primes
            if len(gap_primes) > 1:
                gaps = np.diff(gap_primes)
                normalized = gaps / np.log(gap_primes[:-1])
                normalized_gap_chunks.append(normalized.astype(np.float32))
            previous_prime = int(primes[-1])

        elapsed = time.perf_counter() - block_started
        block_rows.append(
            {
                "block_id": block_id,
                "low": low,
                "high": high,
                "midpoint": math.sqrt(low * high),
                "size": size,
                "prime_count": int(prime_mask.sum()),
                "prime_density": float(prime_mask.mean()),
                "expected_density": float(1.0 / math.log(math.sqrt(low * high))),
                "elapsed_seconds": elapsed,
            }
        )
        print(
            f"    primes={int(prime_mask.sum()):,}  aggregate psi ratio="
            f"{weighted_observed.sum() / weighted_predicted.sum():.6f}  time={elapsed:.1f}s",
            flush=True,
        )
        del prime_mask, prime_indices, primes, observed_count, weighted_observed, null_count, frame
        gc.collect()

    pair_metrics = pd.concat(pair_frames, ignore_index=True)
    pair_metrics.to_csv(results_dir / "pair_metrics.csv.gz", index=False, compression="gzip")
    block_summary = pd.DataFrame(block_rows)
    block_summary.to_csv(results_dir / "block_summary.csv", index=False)

    if normalized_gap_chunks:
        normalized_gaps = np.concatenate(normalized_gap_chunks)
        edges = np.concatenate((np.linspace(0.0, 5.0, 101), [np.inf]))
        counts, _ = np.histogram(normalized_gaps, bins=edges)
        gap_frame = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts})
        gap_frame.to_csv(results_dir / "gap_distribution.csv", index=False)
        gap_summary = {
            "count": int(len(normalized_gaps)),
            "mean": float(normalized_gaps.mean()),
            "variance": float(normalized_gaps.var()),
            "median": float(np.median(normalized_gaps)),
            "p90": float(np.quantile(normalized_gaps, 0.90)),
            "p99": float(np.quantile(normalized_gaps, 0.99)),
            "maximum": float(normalized_gaps.max()),
        }
        (results_dir / "gap_summary.json").write_text(json.dumps(gap_summary, indent=2) + "\n")

    analysis = analyze_results(results_dir)
    total_elapsed = time.perf_counter() - started
    metadata = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "blocks": len(blocks),
        "offsets": len(offsets),
        "total_elapsed_seconds": total_elapsed,
        "results_directory": str(Path(config.results_dir)),
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Completed in {total_elapsed:.1f}s. Results: {results_dir}", flush=True)
    return {"metadata": metadata, "analysis": analysis}
