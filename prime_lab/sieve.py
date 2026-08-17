from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np


def simple_sieve(limit: int) -> np.ndarray:
    """Return all primes p with 2 <= p <= limit."""
    if limit < 2:
        return np.empty(0, dtype=np.int64)
    mask = np.ones(limit + 1, dtype=np.bool_)
    mask[:2] = False
    for p in range(2, math.isqrt(limit) + 1):
        if mask[p]:
            mask[p * p :: p] = False
    return np.flatnonzero(mask).astype(np.int64, copy=False)


def segmented_prime_mask(
    low: int,
    high: int,
    base_primes: np.ndarray | None = None,
) -> np.ndarray:
    """Return a Boolean primality mask for the half-open interval [low, high)."""
    if low < 0 or high < low:
        raise ValueError("require 0 <= low <= high")
    size = high - low
    mask = np.ones(size, dtype=np.bool_)
    if size == 0:
        return mask

    if base_primes is None:
        base_primes = simple_sieve(math.isqrt(max(high - 1, 0)))

    for p_raw in base_primes:
        p = int(p_raw)
        if p * p >= high:
            break
        first = max(p * p, ((low + p - 1) // p) * p)
        if first < high:
            mask[first - low :: p] = False

    if low < 2:
        mask[: min(2 - low, size)] = False
    return mask


def von_mangoldt_segment(
    low: int,
    high: int,
    prime_mask: np.ndarray,
    base_primes: np.ndarray,
) -> np.ndarray:
    """Return Lambda(n) for n in [low, high), including prime powers."""
    if len(prime_mask) != high - low:
        raise ValueError("prime_mask length does not match the interval")

    weights = np.zeros(high - low, dtype=np.float64)
    prime_indices = np.flatnonzero(prime_mask)
    if len(prime_indices):
        weights[prime_indices] = np.log(low + prime_indices)

    # Every p^k with k >= 2 and p^k < high has p <= sqrt(high - 1).
    for p_raw in base_primes:
        p = int(p_raw)
        power = p * p
        if power >= high:
            break
        log_p = math.log(p)
        while power < high:
            if power >= low:
                weights[power - low] = log_p
            if power > (high - 1) // p:
                break
            power *= p
    return weights


def geometric_blocks(start: int, stop: int, ratio: float) -> Iterator[tuple[int, int]]:
    """Yield approximately log-spaced half-open blocks covering [start, stop)."""
    if not (2 <= start < stop):
        raise ValueError("require 2 <= start < stop")
    if ratio <= 1.0:
        raise ValueError("ratio must exceed 1")

    low = int(start)
    while low < stop:
        high = min(stop, max(low + 1, int(round(low * ratio))))
        yield low, high
        low = high

