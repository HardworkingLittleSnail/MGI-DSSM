from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import r2_score


def rul_value_error(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Tuple[int, int, int, float]:
    """PatchFormer-compatible RUL error calculation."""

    true_re, pred_re = len(y_true), 0
    for i in range(len(y_true) - 1):
        if y_true[i] <= threshold >= y_true[i + 1]:
            true_re = i - 1
            break
    for i in range(len(y_pred) - 1):
        if y_pred[i] <= threshold:
            pred_re = i - 1
            break
    rul_real = true_re + 1
    rul_pred = pred_re + 1
    ae_error = abs(true_re - pred_re)
    denom = max(abs(true_re), 1)
    re_score = min(abs(true_re - pred_re) / denom, 1)
    return int(rul_real), int(rul_pred), int(ae_error), float(re_score)


def patchformer_capacity_metrics(y_true_ah: np.ndarray, y_pred_ah: np.ndarray, rated_capacity: float) -> Dict[str, float]:
    """Compute the same capacity metrics used by PatchFormer's RUL script."""

    y_true = np.asarray(y_true_ah, dtype=np.float64)
    y_pred = np.asarray(y_pred_ah, dtype=np.float64)
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    mask = y_true >= 0.0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "rul_real": float("nan"),
            "rul_pred": float("nan"),
            "ae": float("nan"),
            "re": float("nan"),
        }

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    rul_real, rul_pred, ae, re = rul_value_error(
        y_true,
        y_pred,
        threshold=float(rated_capacity) * 0.7,
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "rul_real": float(rul_real),
        "rul_pred": float(rul_pred),
        "ae": float(ae),
        "re": float(re),
    }
