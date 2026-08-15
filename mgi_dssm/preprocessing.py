from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IsolatedSigmaResult:
    values: np.ndarray
    local_mean: np.ndarray
    local_std: np.ndarray
    z_score: np.ndarray
    sigma_candidate: np.ndarray
    isolated: np.ndarray
    missing: np.ndarray
    repaired: np.ndarray


def isolated_sigma_interpolate(
    values: np.ndarray,
    window: int = 21,
    sigma: float = 3.0,
    min_neighbours: int = 6,
    preserve_endpoints: bool = True,
) -> IsolatedSigmaResult:
    """BATTER-MoE-style isolated-outlier removal and linear interpolation.

    Candidate detection is a single pass over the immutable source sequence.
    Only a one-point candidate run is repaired; adjacent candidate runs are
    retained as local fluctuations. Missing internal points are interpolated
    separately as data-integrity repairs. No endpoint extrapolation is used.
    """

    source = np.asarray(values, dtype=np.float64).copy()
    n = len(source)
    cleaned = source.copy()
    local_mean = np.full(n, np.nan, dtype=np.float64)
    local_std = np.full(n, np.nan, dtype=np.float64)
    z_score = np.full(n, np.nan, dtype=np.float64)
    candidate = np.zeros(n, dtype=bool)
    missing = ~np.isfinite(source)

    if window < 5 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 5")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if min_neighbours < 2:
        raise ValueError("min_neighbours must be at least 2")

    radius = window // 2
    for index, value in enumerate(source):
        if not np.isfinite(value):
            continue
        if preserve_endpoints and index in {0, n - 1}:
            continue
        left = max(0, index - radius)
        right = min(n, index + radius + 1)
        neighbours = np.concatenate((source[left:index], source[index + 1 : right]))
        neighbours = neighbours[np.isfinite(neighbours)]
        if neighbours.size < min_neighbours:
            continue
        mean = float(neighbours.mean())
        std = float(neighbours.std(ddof=1))
        local_mean[index] = mean
        local_std[index] = std
        if std <= 0.0:
            continue
        z = abs(value - mean) / std
        z_score[index] = z
        candidate[index] = z > sigma

    candidate_left = np.r_[False, candidate[:-1]]
    candidate_right = np.r_[candidate[1:], False]
    isolated = candidate & ~candidate_left & ~candidate_right

    # Missing values are not labelled as sigma outliers. They are repaired as
    # a separate integrity issue when bounded by two observed/re retained data.
    repair_requested = isolated | missing
    valid = np.flatnonzero(~repair_requested & np.isfinite(source))
    repaired = np.zeros(n, dtype=bool)
    if valid.size >= 2:
        internal = np.flatnonzero(
            repair_requested
            & (np.arange(n) > valid[0])
            & (np.arange(n) < valid[-1])
        )
        if internal.size:
            cleaned[internal] = np.interp(internal, valid, source[valid])
            repaired[internal] = True

    return IsolatedSigmaResult(
        values=cleaned,
        local_mean=local_mean,
        local_std=local_std,
        z_score=z_score,
        sigma_candidate=candidate,
        isolated=isolated,
        missing=missing,
        repaired=repaired,
    )


def fit_train_minmax(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature-wise min/max using training values only."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("values must have shape [samples, features]")
    if not np.isfinite(array).all():
        raise ValueError("training values contain non-finite entries")
    minimum = array.min(axis=0)
    maximum = array.max(axis=0)
    scale = maximum - minimum
    scale[scale < 1e-12] = 1.0
    return minimum.astype(np.float32), scale.astype(np.float32)


def capacity_soh(capacity_ah: np.ndarray, rated_capacity_ah: float) -> np.ndarray:
    """Normalize capacity exactly as C/C0, with C0 the rated capacity."""

    rated = float(rated_capacity_ah)
    if not np.isfinite(rated) or rated <= 0.0:
        raise ValueError("rated_capacity_ah must be positive and finite")
    return np.asarray(capacity_ah, dtype=np.float64) / rated
