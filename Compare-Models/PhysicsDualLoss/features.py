from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


RAW_FEATURES = ("CCCT", "CVCT", "CCDT", "Resistance")
COMPONENTS = ("Trend", "Seasonal", "Residual")


def _duration(time: np.ndarray, mask: np.ndarray) -> float:
    indices = np.flatnonzero(mask)
    if len(indices) < 2:
        return float("nan")
    return float(time[indices[-1]] - time[indices[0]])


def _charge_times(record: object) -> tuple[float, float]:
    time = np.asarray(record.data.Time, dtype=np.float64).reshape(-1)
    current = np.asarray(record.data.Current_measured, dtype=np.float64).reshape(-1)
    voltage = np.asarray(record.data.Voltage_measured, dtype=np.float64).reshape(-1)
    size = min(len(time), len(current), len(voltage))
    time, current, voltage = time[:size], current[:size], voltage[:size]
    valid = np.isfinite(time) & np.isfinite(current) & np.isfinite(voltage)
    positive = valid & (current > 0.02)
    if positive.sum() < 3:
        return float("nan"), float("nan")
    plateau = float(np.median(current[positive & (current > np.quantile(current[positive], 0.6))]))
    # NASA's nominal CC current is 1.5 A. The measured-voltage trace stays a
    # little below 4.2 V, so the CC/CV transition is robustly identified by
    # the first post-4.0 V fall below 95% of the observed current plateau.
    transitions = np.flatnonzero(positive & (voltage >= 4.0) & (current < 0.95 * plateau))
    start = int(np.flatnonzero(positive)[0])
    transition = int(transitions[0]) if len(transitions) else int(np.flatnonzero(positive)[-1])
    end = int(np.flatnonzero(positive)[-1])
    return max(float(time[transition] - time[start]), 0.0), max(float(time[end] - time[transition]), 0.0)


def _discharge_time(record: object) -> float:
    time = np.asarray(record.data.Time, dtype=np.float64).reshape(-1)
    current = np.asarray(record.data.Current_measured, dtype=np.float64).reshape(-1)
    size = min(len(time), len(current))
    return _duration(time[:size], np.isfinite(time[:size]) & np.isfinite(current[:size])
                     & (current[:size] < -0.1))


def extract_nasa_health_features(raw_directory: Path,
                                 summaries: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Extract the paper's four CALCE-type HFs from official NASA records."""
    result: dict[str, pd.DataFrame] = {}
    for battery, summary in summaries.items():
        structure = loadmat(raw_directory / f"{battery}.mat", squeeze_me=True,
                            struct_as_record=False)[battery]
        records = np.atleast_1d(structure.cycle)
        latest_resistance = float("nan")
        latest_charge = (float("nan"), float("nan"))
        by_record: dict[int, tuple[float, float, float, float]] = {}
        for index, record in enumerate(records):
            operation = str(record.type).strip().lower()
            if operation == "impedance":
                values = np.asarray(record.data.Re, dtype=np.float64).reshape(-1)
                values = values[np.isfinite(values) & (values > 0)]
                if len(values):
                    latest_resistance = float(np.median(values))
            elif operation == "charge":
                latest_charge = _charge_times(record)
            elif operation == "discharge":
                by_record[index] = (*latest_charge, _discharge_time(record), latest_resistance)
        frame = summary.copy().sort_values("Cycle").reset_index(drop=True)
        rows = [by_record.get(int(index), (np.nan,) * 4)
                for index in frame["raw_record_index"]]
        values = pd.DataFrame(rows, columns=RAW_FEATURES, dtype=np.float64)
        # Only earlier observations from the same cell may fill a missing HF.
        values = values.ffill().bfill(limit=1)
        for column in RAW_FEATURES:
            if values[column].isna().any():
                values[column] = values[column].fillna(values[column].median())
        result[battery] = pd.concat(
            [frame[["BatteryName", "Cycle", "Capacity"]].reset_index(drop=True), values], axis=1
        )
    return result


def extract_summary_health_features(
    dataset: str, summaries: dict[str, pd.DataFrame]
) -> tuple[dict[str, pd.DataFrame], tuple[str, ...]]:
    """Build the paper's HFs from an immutable aligned summary.

    CALCE exposes CCCT/CVCT/resistance directly and uses 1.1 A CC
    discharge; TJU exposes charge times and uses 2.5 A CC discharge.
    Therefore CCDT = 3600*Capacity/I is physically identical to the
    recorded constant-current duration under the paper's test protocol.
    """
    if dataset not in {"calce", "tju"}:
        raise ValueError(f"summary HF adapter does not support {dataset!r}")
    raw_features = RAW_FEATURES if dataset == "calce" else RAW_FEATURES[:3]
    discharge_current = 1.1 if dataset == "calce" else 2.5
    output: dict[str, pd.DataFrame] = {}
    for battery, source in summaries.items():
        frame = source.copy().sort_values("Cycle").reset_index(drop=True)
        if dataset == "calce":
            values = pd.DataFrame({
                "CCCT": pd.to_numeric(frame["CCCT"], errors="coerce"),
                "CVCT": pd.to_numeric(frame["CVCT"], errors="coerce"),
                "CCDT": 3600.0 * pd.to_numeric(frame["Capacity"], errors="coerce")
                        / discharge_current,
                "Resistance": pd.to_numeric(frame["Resistance"], errors="coerce"),
            })
        else:
            values = pd.DataFrame({
                "CCCT": pd.to_numeric(frame["CC charge time"], errors="coerce"),
                "CVCT": pd.to_numeric(frame["CV charge time"], errors="coerce"),
                "CCDT": 3600.0 * pd.to_numeric(frame["Capacity"], errors="coerce")
                        / discharge_current,
            })
        values = values.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        if values.isna().any().any():
            values = values.fillna(values.median(numeric_only=True)).fillna(0.0)
        output[battery] = pd.concat([
            frame[["BatteryName", "Cycle", "Capacity"]].reset_index(drop=True),
            values[list(raw_features)].reset_index(drop=True),
        ], axis=1)
    return output, raw_features


def _loess_endpoint(values: np.ndarray, end: int, span: int) -> float:
    begin = max(0, end - span + 1)
    y = np.asarray(values[begin:end + 1], dtype=np.float64)
    x = np.arange(begin, end + 1, dtype=np.float64)
    if len(y) <= 2:
        return float(y[-1])
    distance = (x[-1] - x) / max(x[-1] - x[0], 1.0)
    weights = np.power(np.clip(1.0 - distance ** 3, 0.0, None), 3)
    design = np.column_stack((np.ones_like(x), x - x[-1]))
    beta = np.linalg.pinv(design.T @ (weights[:, None] * design)) @ (
        design.T @ (weights * y)
    )
    return float(beta[0])


def detect_period(values: np.ndarray, maximum_period: int = 32) -> int:
    """Paper equations (5)-(6), with disclosed gaps fixed deterministically."""
    values = np.asarray(values, dtype=np.float64)
    preliminary = pd.Series(values).rolling(7, min_periods=1).mean().to_numpy()
    detrended = values - preliminary
    maximum = min(maximum_period, max(2, len(values) // 4))
    correlations = np.asarray([
        np.corrcoef(detrended[:-lag], detrended[lag:])[0, 1]
        if np.std(detrended[:-lag]) > 0 and np.std(detrended[lag:]) > 0 else 0.0
        for lag in range(1, maximum + 1)
    ])
    peaks, _ = find_peaks(np.nan_to_num(correlations, nan=0.0))
    candidates = {int(index + 1) for index in peaks}
    candidates.update(period for period in (5, 7, 10, 14, 21, 28) if period <= maximum)
    if not candidates:
        return min(7, maximum)
    best_period, best_score = min(candidates), -float("inf")
    for period in sorted(candidates):
        trend = pd.Series(values).rolling(2 * period + 1, min_periods=1).mean().to_numpy()
        residual = values - trend
        seasonal = np.asarray([
            np.mean(residual[np.arange(len(values)) % period == phase])
            for phase in range(period)
        ])[np.arange(len(values)) % period]
        noise = residual - seasonal
        strength = np.std(seasonal) / max(np.std(noise), 1e-8)
        smoothness = 1.0 - np.mean(np.abs(np.diff(trend))) / max(np.std(values), 1e-8)
        score = 0.5 * strength + 0.5 * smoothness
        if score > best_score:
            best_period, best_score = period, score
    return int(best_period)


def causal_stl(values: np.ndarray, period: int) -> np.ndarray:
    """Online STL adaptation: Loess trend and historical phase seasonality only."""
    values = np.asarray(values, dtype=np.float64)
    trend = np.empty_like(values)
    seasonal = np.zeros_like(values)
    residual = np.empty_like(values)
    phase_history: list[list[float]] = [[] for _ in range(period)]
    span = max(5, 2 * period + 1)
    for index in range(len(values)):
        trend[index] = _loess_endpoint(values, index, span)
        phase = index % period
        seasonal[index] = float(np.mean(phase_history[phase])) if phase_history[phase] else 0.0
        residual[index] = values[index] - trend[index] - seasonal[index]
        phase_history[phase].append(values[index] - trend[index])
    return np.column_stack((trend, seasonal, residual))


def decompose_features(frames: dict[str, pd.DataFrame], period: int,
                       relative_to_initial: bool = True,
                       raw_features: tuple[str, ...] = RAW_FEATURES) -> tuple[dict[str, np.ndarray], list[str]]:
    names = [f"{feature}{component}" for feature in raw_features for component in COMPONENTS]
    output = {
        battery: np.concatenate([
            causal_stl(
                frame[feature].to_numpy(dtype=np.float64)
                / max(abs(float(frame[feature].iloc[0])), 1e-8)
                if relative_to_initial else frame[feature].to_numpy(dtype=np.float64),
                period,
            )
            for feature in raw_features
        ], axis=1).astype(np.float32)
        for battery, frame in frames.items()
    }
    return output, names


@dataclass(frozen=True)
class FeatureSelection:
    indices: tuple[int, ...]
    names: tuple[str, ...]
    random_forest: tuple[str, ...]
    lasso: tuple[str, ...]
    voted: tuple[str, ...]


def select_wrapper_features(decomposed: dict[str, np.ndarray], frames: dict[str, pd.DataFrame],
                            train_names: list[str], feature_names: list[str],
                            split_indices: dict[str, int], seed: int) -> FeatureSelection:
    matrix, target, groups = [], [], []
    for group_index, name in enumerate(train_names):
        stop = split_indices[name]
        matrix.append(decomposed[name][:stop - 1])
        target.append(frames[name]["Capacity"].to_numpy(dtype=np.float64)[1:stop])
        groups.append(np.full(stop - 1, group_index, dtype=np.int64))
    x = np.concatenate(matrix)
    y = np.concatenate(target)
    group = np.concatenate(groups)
    forest = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2,
                                   min_samples_leaf=1, random_state=seed, n_jobs=1).fit(x, y)
    rf_indices = np.argsort(forest.feature_importances_)[::-1][:6]
    scaled = StandardScaler().fit_transform(x)
    lasso = LassoCV(cv=5, random_state=seed, max_iter=20000).fit(scaled, y)
    lasso_indices = np.flatnonzero(np.abs(lasso.coef_) > 1e-10)
    rf_set, lasso_set = set(rf_indices.tolist()), set(lasso_indices.tolist())
    intersection = rf_set & lasso_set
    voted = intersection if len(rf_set - lasso_set) >= 3 and len(intersection) >= 3 else rf_set | lasso_set
    voted_indices = np.asarray(sorted(voted), dtype=np.int64)
    if len(voted_indices) < 3:
        voted_indices = rf_indices[:3]
    selector = RFECV(
        RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1),
        step=1, min_features_to_select=3, cv=GroupKFold(n_splits=len(train_names)),
        scoring="neg_mean_squared_error", n_jobs=1,
    )
    selector.fit(x[:, voted_indices], y, groups=group)
    final_indices = tuple(int(value) for value in voted_indices[selector.support_])
    return FeatureSelection(
        final_indices, tuple(feature_names[index] for index in final_indices),
        tuple(feature_names[index] for index in rf_indices),
        tuple(feature_names[index] for index in lasso_indices),
        tuple(feature_names[index] for index in voted_indices),
    )
