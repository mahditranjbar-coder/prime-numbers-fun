from __future__ import annotations

import math

import numpy as np


def wheel_candidate_mask(low: int, high: int, wheel_primes: tuple[int, ...]) -> np.ndarray:
    values = np.arange(low, high, dtype=np.int64)
    candidates = np.ones(high - low, dtype=np.bool_)
    for p in wheel_primes:
        candidates &= values % p != 0
    return candidates


def wheel_conditioned_sample(
    low: int,
    high: int,
    wheel_primes: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a varying-density random prime surrogate after wheel filtering."""
    candidates = wheel_candidate_mask(low, high, wheel_primes)
    wheel = math.prod(wheel_primes)
    phi = math.prod(p - 1 for p in wheel_primes)
    values = np.arange(low, high, dtype=np.float64)
    conditional_probability = (wheel / phi) / np.log(values)
    conditional_probability = np.minimum(conditional_probability, 1.0)
    draws = rng.random(high - low)
    return candidates & (draws < conditional_probability)

