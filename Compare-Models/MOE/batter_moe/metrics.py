from __future__ import annotations

import numpy as np


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = target - prediction
    denominator = np.square(target - target.mean()).sum()
    return {
        'mae': float(np.abs(error).mean()),
        'rmse': float(np.sqrt(np.square(error).mean())),
        'r2': float(1 - np.square(error).sum() / denominator) if denominator > 0 else float('nan'),
    }


def first_eol_cycle(cycles: np.ndarray, capacity: np.ndarray, threshold: float) -> int:
    below = np.flatnonzero(np.asarray(capacity) < threshold)
    # A trajectory not crossing before recording stops is right-censored at
    # one cycle after the observation horizon. This policy is explicit because
    # the paper does not state how it handles non-crossing predictions.
    return int(cycles[below[0]]) if len(below) else int(cycles[-1] + 1)


def evaluate_from_start(
    cycles: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    start_point: int,
    eol_fraction: float,
) -> dict[str, float]:
    selected = np.asarray(cycles) >= start_point
    result = regression_metrics(np.asarray(target)[selected], np.asarray(prediction)[selected])
    true_eol = first_eol_cycle(cycles, target, eol_fraction)
    pred_eol = first_eol_cycle(cycles, prediction, eol_fraction)
    true_rul, pred_rul = true_eol - start_point, pred_eol - start_point
    result.update({
        'true_eol_cycle': true_eol,
        'predicted_eol_cycle': pred_eol,
        'true_rul': true_rul,
        'predicted_rul': pred_rul,
        're': abs(true_rul - pred_rul) / true_rul if true_rul > 0 else float('nan'),
    })
    return result
