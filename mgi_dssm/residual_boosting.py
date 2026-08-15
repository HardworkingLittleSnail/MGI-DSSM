from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from .data import battery_names, linear_extrapolation
from .metrics import patchformer_capacity_metrics


@dataclass
class ResidualBoostingConfig:
    seq_len: int = 64
    max_seq_len: int = 1000
    start_points: Tuple[int, ...] = (300, 400, 500)
    seed: int = 7
    max_iter: int = 300
    learning_rate: float = 0.02
    max_leaf_nodes: int = 12
    l2_regularization: float = 0.1
    feature_mode: str = "calce-summary"
    rated_capacity: float = 1.1


def _slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    return float(np.polyfit(x, values.astype(np.float64), deg=1)[0])


def _clean(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    if np.isfinite(values).any():
        fill = float(np.nanmedian(values[np.isfinite(values)]))
    else:
        fill = 0.0
    return np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill)


def window_features(window: pd.DataFrame, feature_mode: str = "capacity") -> np.ndarray:
    """Hand-built non-leaking features for one 64-cycle history window."""

    out: List[float] = []
    if feature_mode == "capacity":
        columns = ["Capacity"]
    elif feature_mode == "calce-summary":
        columns = ["Capacity", "Resistance", "CCCT", "CVCT"]
    else:
        raise ValueError(f"Unknown residual feature mode: {feature_mode}")

    for col in columns:
        arr = _clean(window[col].to_numpy(dtype=np.float64))
        out.extend(
            [
                float(arr[-1]),
                float(arr[-1] - arr[0]),
                float(arr[-1] - np.mean(arr[-5:])),
                float(arr[-1] - np.median(arr[-5:])),
            ]
        )
        for k in [3, 5, 10, 20, 64]:
            z = arr[-k:]
            out.extend(
                [
                    float(np.mean(z)),
                    float(np.median(z)),
                    float(np.std(z)),
                    _slope(z),
                    float(z[-1] - z[0]),
                ]
            )
        d = np.diff(arr)
        for k in [3, 5, 10, 20, 63]:
            dz = d[-k:]
            out.extend(
                [
                    float(np.mean(dz)),
                    float(np.median(dz)),
                    float(np.std(dz)),
                    float(np.min(dz)),
                    float(np.max(dz)),
                ]
            )
        out.extend([float(v) for v in arr[-5:]])
        out.extend([float(v) for v in d[-5:]])

    cap = _clean(window["Capacity"].to_numpy(dtype=np.float64))
    out.append(float(cap[-1] / max(cap[0], 1e-6)))
    if feature_mode == "calce-summary":
        res = _clean(window["Resistance"].to_numpy(dtype=np.float64))
        ccct = _clean(window["CCCT"].to_numpy(dtype=np.float64))
        cvct = _clean(window["CVCT"].to_numpy(dtype=np.float64))
        out.extend(
            [
                float(res[-1] / max(res[0], 1e-6)),
                float(ccct[-1] / max(ccct[0], 1e-6)),
                float(cvct[-1] / max(cvct[0], 1e-6)),
            ]
        )
    out.extend(
        [
            float(window.iloc[-1]["Cycle"]) / 1000.0,
            float(np.log1p(float(window.iloc[-1]["Cycle"])) / np.log1p(1000.0)),
        ]
    )
    return np.nan_to_num(np.asarray(out, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def build_window_arrays(
    frame: pd.DataFrame,
    batteries: Iterable[str],
    seq_len: int,
    max_seq_len: int,
    start_point: int | None = None,
    feature_mode: str = "capacity",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    xs: List[np.ndarray] = []
    residuals: List[float] = []
    targets: List[float] = []
    lasts: List[float] = []
    cycles: List[int] = []
    battery_ids: List[str] = []

    for battery in batteries:
        group = frame[frame["BatteryName"] == battery].sort_values("Cycle").reset_index(drop=True)
        group = group[group["Cycle"] <= int(max_seq_len)].reset_index(drop=True)
        for pos in range(seq_len, len(group)):
            cycle = int(group.iloc[pos]["Cycle"])
            if start_point is not None and cycle < int(start_point):
                continue
            window = group.iloc[pos - seq_len : pos]
            target = float(group.iloc[pos]["Capacity"])
            last = float(window.iloc[-1]["Capacity"])
            xs.append(window_features(window, feature_mode=feature_mode))
            residuals.append(target - last)
            targets.append(target)
            lasts.append(last)
            cycles.append(cycle)
            battery_ids.append(str(battery))

    if not xs:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [],
        )
    return (
        np.vstack(xs).astype(np.float32),
        np.asarray(residuals, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(lasts, dtype=np.float32),
        np.asarray(cycles, dtype=np.int64),
        battery_ids,
    )


def _make_model(config: ResidualBoostingConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=int(config.max_iter),
        learning_rate=float(config.learning_rate),
        max_leaf_nodes=int(config.max_leaf_nodes),
        l2_regularization=float(config.l2_regularization),
        random_state=int(config.seed),
    )


def run_residual_boosting(
    frame: pd.DataFrame,
    config: ResidualBoostingConfig,
    output_dir: Path,
    test_batteries: Iterable[str] | None = None,
) -> Dict[str, object]:
    selected = list(test_batteries) if test_batteries else battery_names(frame)
    all_batteries = battery_names(frame)
    metrics: List[Dict[str, object]] = []
    predictions: List[Dict[str, object]] = []

    for test_battery in selected:
        train_batteries = [name for name in all_batteries if name != test_battery]
        x_train, y_train, _, _, _, _ = build_window_arrays(
            frame,
            train_batteries,
            seq_len=config.seq_len,
            max_seq_len=config.max_seq_len,
            feature_mode=config.feature_mode,
        )
        model = _make_model(config)
        model.fit(x_train, y_train)

        for start_point in config.start_points:
            x, y_res, y_true, last, cycles, battery_ids = build_window_arrays(
                frame,
                [test_battery],
                seq_len=config.seq_len,
                max_seq_len=config.max_seq_len,
                start_point=int(start_point),
                feature_mode=config.feature_mode,
            )
            pred_res = model.predict(x).astype(np.float64)
            pred = last.astype(np.float64) + pred_res
            persistence = last.astype(np.float64)
            linear = []
            group = frame[frame["BatteryName"] == test_battery].sort_values("Cycle").reset_index(drop=True)
            for cycle in cycles:
                hist = group[group["Cycle"].between(int(cycle) - config.seq_len, int(cycle) - 1)]
                linear.append(linear_extrapolation(hist["Capacity"].to_numpy(dtype=np.float32), window=5))
            linear_arr = np.asarray(linear, dtype=np.float64)

            main_metrics = patchformer_capacity_metrics(y_true, pred, rated_capacity=config.rated_capacity)
            mae = float(main_metrics["mae"])
            rmse = float(main_metrics["rmse"])
            persistence_mae = float(mean_absolute_error(y_true, persistence))
            linear_mae = float(mean_absolute_error(y_true, linear_arr))
            metrics.append(
                {
                    "fold": f"leave_{test_battery}_out",
                    "test_battery": test_battery,
                    "start_point": int(start_point),
                    "train_batteries": train_batteries,
                    "num_windows": int(len(y_true)),
                    "mae": mae,
                    "rmse": rmse,
                    "r2": float(main_metrics["r2"]),
                    "RUL_real": int(main_metrics["rul_real"]),
                    "RUL_pred": int(main_metrics["rul_pred"]),
                    "AE": int(main_metrics["ae"]),
                    "RE": float(main_metrics["re"]),
                    "persistence_mae": persistence_mae,
                    "linear_last5_mae": linear_mae,
                }
            )

            for i, cycle in enumerate(cycles):
                predictions.append(
                    {
                        "fold": f"leave_{test_battery}_out",
                        "test_battery": test_battery,
                        "start_point": int(start_point),
                        "cycle": int(cycle),
                        "capacity_true": float(y_true[i]),
                        "capacity_pred": float(pred[i]),
                        "residual_pred": float(pred_res[i]),
                        "residual_true": float(y_res[i]),
                        "persistence_pred": float(persistence[i]),
                        "linear_last5_pred": float(linear_arr[i]),
                        "abs_error": float(abs(pred[i] - y_true[i])),
                        "persistence_abs_error": float(abs(persistence[i] - y_true[i])),
                        "linear_last5_abs_error": float(abs(linear_arr[i] - y_true[i])),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    if metrics:
        summary = {
            "mae": float(np.mean([m["mae"] for m in metrics])),
            "rmse": float(np.mean([m["rmse"] for m in metrics])),
            "r2": float(np.nanmean([m["r2"] for m in metrics])),
            "AE": float(np.nanmean([m["AE"] for m in metrics])),
            "RE": float(np.nanmean([m["RE"] for m in metrics])),
            "persistence_mae": float(np.mean([m["persistence_mae"] for m in metrics])),
            "linear_last5_mae": float(np.mean([m["linear_last5_mae"] for m in metrics])),
            "num_fold_startpoint_runs": len(metrics),
            "num_prediction_windows": len(predictions),
        }
    else:
        summary = {}

    payload: Dict[str, object] = {
        "protocol": "patchformer",
        "head": "residual_hgb",
        "task": f"past_{config.seq_len}_cycles_to_next_capacity",
        "preprocessing": {
            "macro_indicator_imputation": "per-cell causal forward fill; leading missing values become 0.0",
            "train_test_split": "leave-one-battery-out; held-out battery excluded from model fitting",
            "window_policy": "history cycles [target_cycle-seq_len, target_cycle-1] only",
        },
        "config": asdict(config),
        "folds": metrics,
        "metrics": summary,
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if predictions:
        with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
            writer.writeheader()
            writer.writerows(predictions)
    return payload


def format_residual_results(payload: Dict[str, object]) -> str:
    rows = payload["folds"]
    metrics = payload["metrics"]
    lines = []
    header = (
        "battery".ljust(12)
        + " start".rjust(8)
        + " windows".rjust(10)
        + " mae".rjust(12)
        + " rmse".rjust(12)
        + " r2".rjust(10)
        + " AE".rjust(8)
        + " RE".rjust(10)
        + " persist".rjust(12)
        + " linear5".rjust(12)
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:  # type: ignore[assignment]
        lines.append(
            str(row["test_battery"]).ljust(12)
            + f"{row['start_point']:8d}"
            + f"{row['num_windows']:10d}"
            + f"{row['mae']:12.6f}"
            + f"{row['rmse']:12.6f}"
            + f"{row['r2']:10.4f}"
            + f"{row['AE']:8d}"
            + f"{row['RE']:10.4f}"
            + f"{row['persistence_mae']:12.6f}"
            + f"{row['linear_last5_mae']:12.6f}"
        )
    lines.append("")
    lines.append(
        "Overall: "
        f"MAE={metrics['mae']:.6f} Ah, "
        f"RMSE={metrics['rmse']:.6f} Ah, "
        f"R2={metrics['r2']:.4f}, "
        f"AE={metrics['AE']:.2f}, "
        f"RE={metrics['RE']:.4f}, "
        f"Persistence MAE={metrics['persistence_mae']:.6f} Ah, "
        f"LinearLast5 MAE={metrics['linear_last5_mae']:.6f} Ah"
    )
    return "\n".join(lines)
