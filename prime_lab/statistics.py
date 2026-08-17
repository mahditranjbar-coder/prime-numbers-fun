from __future__ import annotations

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.integrate import quad


def autocorrelation_lags(
    values: np.ndarray,
    max_lag: int,
    *,
    workers: int = -1,
) -> np.ndarray:
    """Compute sum_i values[i]*values[i+h] for 0 <= h <= max_lag.

    Padding by max_lag is sufficient because only those lags are requested; the
    zero tail prevents circular wraparound over this range.
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if max_lag < 0 or max_lag >= len(values):
        raise ValueError("require 0 <= max_lag < len(values)")

    fft_length = next_fast_len(len(values) + max_lag + 1)
    spectrum = rfft(values, n=fft_length, workers=workers)
    spectrum *= np.conjugate(spectrum)
    correlation = irfft(spectrum, n=fft_length, workers=workers)
    return correlation[: max_lag + 1]


def inverse_log_square_integral(low: int, high: int) -> float:
    """Numerically evaluate integral_low^high dt/log(t)^2."""
    if low <= 1 or high <= low:
        raise ValueError("require 1 < low < high")
    value, _ = quad(lambda t: 1.0 / np.log(t) ** 2, low, high, epsabs=1e-7, epsrel=1e-11)
    return float(value)


def direct_autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    """Slow reference implementation used by tests."""
    values = np.asarray(values)
    return np.array(
        [np.dot(values[: len(values) - h], values[h:]) for h in range(max_lag + 1)],
        dtype=np.float64,
    )

