"""Shared, leakage-safe protocol for all paper comparison models.

The module deliberately owns only the dataset/task contract.  Every baseline
keeps its paper architecture, native input representation, normalization and
training loss in its own runner.
"""
from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = (7, 17, 27, 37, 47, 57, 67, 77, 87, 97)


@dataclass(frozen=True)
class DatasetProtocol:
    name: str
    summary: str
    batteries: tuple[str, ...]
    rated_capacity: float
    eol_fraction: float
    seq_len: int
    start_points: tuple[int, int]

    @property
    def summary_path(self) -> Path:
        return ROOT / self.summary


PROTOCOLS = {
    "nasa": DatasetProtocol(
        "nasa", "data/version3/NASA data/NASA_Data_minimal_interpolated.npy",
        ("B0005", "B0006", "B0007", "B0018"), 2.0, 0.70, 16, (50, 90),
    ),
    "calce": DatasetProtocol(
        "calce", "data/version3/CALCE data/CALCE_Data.npy",
        ("CS2_35", "CS2_36", "CS2_37", "CS2_38"), 1.1, 0.70, 64, (200, 400),
    ),
    "tju": DatasetProtocol(
        "tju", "data/version3/TJU data/TJU_Data_version2_model_adapter.npy",
        ("CY25-1", "CY25-2", "CY25-3"), 2.5, 0.70, 64, (200, 400),
    ),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_summary(protocol: DatasetProtocol) -> dict[str, pd.DataFrame]:
    if not protocol.summary_path.exists():
        raise FileNotFoundError(f"processed summary not found: {protocol.summary_path}")
    loaded = np.load(protocol.summary_path, allow_pickle=True)
    payload = loaded.item() if loaded.shape == () else loaded[0]
    missing = set(protocol.batteries) - set(payload)
    if missing:
        raise KeyError(f"{protocol.name} is missing batteries: {sorted(missing)}")
    result: dict[str, pd.DataFrame] = {}
    for name in protocol.batteries:
        frame = payload[name].copy().sort_values("Cycle").reset_index(drop=True)
        required = {"Cycle", "Capacity"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} lacks columns {sorted(required - set(frame.columns))}")
        if frame["Cycle"].duplicated().any() or not frame["Cycle"].is_monotonic_increasing:
            raise ValueError(f"{name} has invalid cycle ordering")
        if not np.isfinite(frame["Capacity"].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} has non-finite processed capacities")
        result[name] = frame
    return result


def chronological_samples(
    frames: dict[str, pd.DataFrame], names: Iterable[str], seq_len: int,
    validation_fraction: float = 0.2,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Split each training cell chronologically, then form one-step windows."""
    train: list[tuple[str, int]] = []
    validation: list[tuple[str, int]] = []
    for name in names:
        length = len(frames[name])
        split = int(math.floor(length * (1.0 - validation_fraction)))
        if split <= seq_len or length - split < 1:
            raise ValueError(f"{name} is too short for seq_len={seq_len} and validation split")
        train.extend((name, target) for target in range(seq_len, split))
        validation.extend((name, target) for target in range(split, length))
    return train, validation


def first_threshold_index(values: np.ndarray, threshold: float, consecutive: bool = True) -> int:
    values = np.asarray(values, dtype=np.float64)
    if consecutive:
        for index in range(len(values) - 1):
            if values[index] <= threshold and values[index + 1] <= threshold:
                return index - 1
    else:
        hits = np.flatnonzero(values <= threshold)
        if len(hits):
            return int(hits[0] - 1)
    return len(values)


def capacity_metrics(y_true: np.ndarray, y_pred: np.ndarray, rated_capacity: float) -> dict[str, float | int]:
    """Metrics identical to the main model's final one-step protocol."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true >= 0)
    y_true, y_pred = y_true[finite], y_pred[finite]
    if not len(y_true):
        return {key: float("nan") for key in ("MAE", "RMSE", "R2", "RUL_real", "RUL_pred", "AE", "RE")}
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
    denominator = float(np.sum(np.square(y_true - y_true.mean())))
    r2 = 1.0 - float(np.sum(np.square(y_true - y_pred))) / denominator if denominator > 0 else float("nan")
    threshold = float(rated_capacity) * 0.70
    true_index = first_threshold_index(y_true, threshold, consecutive=True)
    pred_index = first_threshold_index(y_pred, threshold, consecutive=True)
    ae = abs(true_index - pred_index)
    re = min(ae / max(abs(true_index), 1), 1.0)
    return {
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "RUL_real": int(true_index + 1), "RUL_pred": int(pred_index + 1),
        "AE": int(ae), "RE": float(re),
    }


def evaluate_predictions(
    cycles: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
    protocol: DatasetProtocol,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = [
        {"cycle": int(c), "capacity_true": float(t), "capacity_pred": float(p), "abs_error": abs(float(t) - float(p))}
        for c, t, p in zip(cycles, y_true, y_pred)
    ]
    metrics: list[dict[str, object]] = []
    for start in protocol.start_points:
        # SP denotes the number of observed cycles; evaluation begins at SP + 1.
        # The benchmark start point is the first predicted target cycle.  This
        # matches our model and the native PatchFormer/RUL-Mamba evaluators.
        mask = np.asarray(cycles) >= start
        item: dict[str, object] = {
            "stage": "early" if start == protocol.start_points[0] else "late",
            "start_point": start, "num_windows": int(mask.sum()),
        }
        item.update(capacity_metrics(np.asarray(y_true)[mask], np.asarray(y_pred)[mask], protocol.rated_capacity))
        metrics.append(item)
    return metrics, rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def protocol_manifest() -> dict[str, object]:
    return {
        "seeds": list(DEFAULT_SEEDS),
        "prediction": "rolling one-step from ground-truth historical windows",
        "split": "leave-one-cell-out; remaining cells chronological 80/20 train/validation",
        "preprocessing": "shared BATTER-MoE-aligned processed capacity summaries; no baseline re-cleaning",
        "evaluation_cells": {name: value.batteries[0] for name, value in PROTOCOLS.items()},
        "datasets": {name: asdict(value) for name, value in PROTOCOLS.items()},
    }
