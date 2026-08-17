from __future__ import annotations

import numpy as np

from .sieve import simple_sieve


# Twin-prime constant. The prime-pair singular series begins with 2*C2.
TWIN_PRIME_CONSTANT = 0.660161815846869573927812110014555778


def even_offsets(max_gap: int) -> np.ndarray:
    if max_gap < 2:
        raise ValueError("max_gap must be at least 2")
    return np.arange(2, max_gap + 1, 2, dtype=np.int64)


def pair_singular_series(offsets: np.ndarray) -> np.ndarray:
    """Hardy-Littlewood singular series for fixed positive pair offsets."""
    offsets = np.asarray(offsets, dtype=np.int64)
    if np.any(offsets <= 0):
        raise ValueError("offsets must be positive")

    result = np.zeros(len(offsets), dtype=np.float64)
    even = offsets % 2 == 0
    result[even] = 2.0 * TWIN_PRIME_CONSTANT
    if not np.any(even):
        return result

    primes = simple_sieve(int(offsets.max()))
    for p_raw in primes[primes > 2]:
        p = int(p_raw)
        divisible = even & (offsets % p == 0)
        result[divisible] *= (p - 1.0) / (p - 2.0)
    return result


def offset_arithmetic_features(offsets: np.ndarray) -> dict[str, np.ndarray]:
    """Return interpretable factorization features for each offset."""
    offsets = np.asarray(offsets, dtype=np.int64)
    remaining = offsets.copy()
    omega = np.zeros(len(offsets), dtype=np.int16)
    big_omega = np.zeros(len(offsets), dtype=np.int16)
    largest_factor = np.ones(len(offsets), dtype=np.int64)
    reciprocal_factor_sum = np.zeros(len(offsets), dtype=np.float64)

    for p_raw in simple_sieve(int(np.sqrt(offsets.max())) + 1):
        p = int(p_raw)
        divisible = remaining % p == 0
        if not np.any(divisible):
            continue
        omega[divisible] += 1
        largest_factor[divisible] = p
        reciprocal_factor_sum[divisible] += 1.0 / p
        while np.any(remaining % p == 0):
            hit = remaining % p == 0
            remaining[hit] //= p
            big_omega[hit] += 1

    leftover = remaining > 1
    omega[leftover] += 1
    big_omega[leftover] += 1
    largest_factor[leftover] = np.maximum(largest_factor[leftover], remaining[leftover])
    reciprocal_factor_sum[leftover] += 1.0 / remaining[leftover]

    return {
        "omega": omega,
        "big_omega": big_omega,
        "largest_prime_factor": largest_factor,
        "reciprocal_factor_sum": reciprocal_factor_sum,
    }

