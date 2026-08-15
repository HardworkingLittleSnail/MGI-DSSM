"""BATTER-MoE-private TJU preprocessing from the official raw CSV files.

Feature construction follows the charging-window framework cited by
BATTER-MoE (Wang et al., Nature Communications 2024).  Cleaning follows the
BATTER-MoE paper instead: isolated 3-sigma points are linearly interpolated,
whereas adjacent candidate runs are retained.  No shared dataset cache is
read or overwritten by this module.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy, linregress


FEATURE_COLUMNS = (
    "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
    "voltage slope", "voltage entropy", "current mean", "current std",
    "current kurtosis", "current skewness", "current slope", "current entropy",
    "CC Q", "CC charge time", "CV Q", "CV charge time", "capacity",
)
RAW_FILES = {
    "CY25-1": "CY25-05_1-#1.csv",
    "CY25-2": "CY25-05_1-#2.csv",
    "CY25-3": "CY25-05_1-#3.csv",
}
USECOLS = (
    "time/s", "Ecell/V", "<I>/mA", "Q discharge/mA.h", "Q charge/mA.h",
    "control/V", "control/mA", "cycle number",
)
PREPROCESSING_VERSION = "wang-segmented-full-charge-features+batter-isolated-global3sigma-v6"
CURRENT_ACTIVITY_THRESHOLD_MA = 10.0
ENTROPY_BINS = 20


def _entropy(values: np.ndarray) -> float:
    """Shannon entropy of the 20-bin value histogram used by Wang's files."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        return 0.0
    counts = np.histogram(values, bins=ENTROPY_BINS)[0]
    counts = counts[counts > 0]
    if not len(counts):
        return 0.0
    return float(entropy(counts))


def _slope(time: np.ndarray, values: np.ndarray) -> float:
    """Ordinary-least-squares curve slope used in Wang's processed files."""
    return float(linregress(time, values).slope) if len(values) > 1 else 0.0


def _curve_features(frame: pd.DataFrame, value_column: str, scale: float) -> tuple[float, ...]:
    time = frame["time/s"].to_numpy(dtype=np.float64)
    values = frame[value_column].to_numpy(dtype=np.float64) / scale
    valid = np.isfinite(time) & np.isfinite(values)
    time, values = time[valid], values[valid]
    if len(values) < 4:
        return (float("nan"),) * 6
    return (
        float(values.mean()),
        float(values.std(ddof=0)),
        # pandas uses the bias-corrected Fisher definitions found in the
        # reference TJU cycle-level files.
        float(pd.Series(values).kurt()),
        float(pd.Series(values).skew()),
        _slope(time, values),
        _entropy(values),
    )


def _duration_and_charge(frame: pd.DataFrame) -> tuple[float, float]:
    if len(frame) < 2:
        return float("nan"), float("nan")
    time = frame["time/s"].to_numpy(dtype=np.float64)
    charge = frame["Q charge/mA.h"].to_numpy(dtype=np.float64) / 1000.0
    delta_time = np.diff(time)
    typical_step = float(np.median(delta_time))
    maximum_contiguous_gap = max(60.0, 5.0 * typical_step)
    duration = float(delta_time[delta_time <= maximum_contiguous_gap].sum())
    accumulated = float(charge[-1] - charge[0])
    if duration <= 0 or accumulated < 0:
        return float("nan"), float("nan")
    return accumulated, duration


def extract_cycle_features(frame: pd.DataFrame) -> np.ndarray:
    frame = frame.sort_values("time/s", kind="stable")
    # Wang's TJU cycle-level files compute both voltage and current statistics
    # over the complete active charging trajectory. A small measured-current
    # threshold removes transition/rest records whose command remains positive.
    charge = frame.loc[frame["<I>/mA"] > CURRENT_ACTIVITY_THRESHOLD_MA]
    cc = frame.loc[
        (frame["control/mA"] > 0)
        & (frame["<I>/mA"] > CURRENT_ACTIVITY_THRESHOLD_MA)
    ]
    cv = frame.loc[
        (frame["control/V"] > 0)
        & (frame["<I>/mA"] > CURRENT_ACTIVITY_THRESHOLD_MA)
    ]
    discharge = frame.loc[frame["<I>/mA"] < -CURRENT_ACTIVITY_THRESHOLD_MA]
    if len(charge) < 4 or len(cc) < 4 or len(cv) < 4 or len(discharge) < 4:
        return np.full(17, np.nan, dtype=np.float64)
    voltage = _curve_features(charge, "Ecell/V", 1.0)
    current = _curve_features(charge, "<I>/mA", 1000.0)
    cc_q, cc_time = _duration_and_charge(cc)
    cv_q, cv_time = _duration_and_charge(cv)
    # Integrating the active discharge segment reproduces the cycle capacity
    # used in Wang-derived TJU feature files; Q-discharge max includes boundary
    # samples outside the active-current interval and is systematically higher.
    discharge_time = discharge["time/s"].to_numpy(dtype=np.float64)
    discharge_current = -discharge["<I>/mA"].to_numpy(dtype=np.float64)
    delta_time = np.diff(discharge_time)
    typical_step = float(np.median(delta_time))
    cut = np.flatnonzero(delta_time > max(60.0, 5.0 * typical_step)) + 1
    segments = np.split(np.arange(len(discharge_time)), cut)
    main = max(
        segments,
        key=lambda index: (
            discharge_time[index[-1]] - discharge_time[index[0]] if len(index) > 1 else 0.0
        ),
    )
    capacity = float(
        np.trapezoid(discharge_current[main], discharge_time[main]) / 3_600_000.0
    )
    return np.asarray(voltage + current + (cc_q, cc_time, cv_q, cv_time, capacity))


def _clean_isolated_3sigma(
    raw: np.ndarray, cycles: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """BATTER-MoE isolated 3-sigma removal followed by interpolation.

    BATTER-MoE states a 3-sigma rule but does not define a local window.
    Therefore use the literal feature-wise global mean and sample standard
    deviation within each cell. Only isolated candidates are repaired;
    adjacent candidate runs are retained as local fluctuations.
    """
    frame = pd.DataFrame(raw, columns=FEATURE_COLUMNS, index=cycles)
    audit: list[dict[str, object]] = []
    for column in FEATURE_COLUMNS:
        series = pd.to_numeric(frame[column], errors="coerce")
        missing = ~np.isfinite(series)
        original = series.copy()
        global_mean = float(series.mean(skipna=True))
        global_std = float(series.std(skipna=True, ddof=1))
        candidate = ((series - global_mean).abs() > 3.0 * global_std).fillna(False)
        isolated = candidate & ~candidate.shift(1, fill_value=False) & ~candidate.shift(-1, fill_value=False)
        repair = isolated | missing
        series.loc[repair] = np.nan
        series = series.interpolate(method="index", limit_direction="both")
        frame[column] = series
        for cycle in frame.index[repair]:
            audit.append({
                "cycle": int(cycle), "feature": column,
                "raw_value": float(original.loc[cycle]) if np.isfinite(original.loc[cycle]) else None,
                "mean": global_mean if np.isfinite(global_mean) else None,
                "std": global_std if np.isfinite(global_std) else None,
                "sigma_candidate": bool(candidate.loc[cycle]),
                "missing": bool(missing.loc[cycle]),
                "action": "linear_interpolation",
                "cleaned_value": float(series.loc[cycle]),
            })
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite TJU indicators remain after BATTER-MoE cleaning")
    return values, audit


def build(root: Path, force: bool = False) -> Path:
    private_dir = Path(__file__).resolve().parent / "private_data"
    output_path = private_dir / "tju_17_features_batter_moe.npy"
    metadata_path = private_dir / "tju_17_features_batter_moe.json"
    audit_path = private_dir / "tju_17_features_batter_moe_audit.csv"
    if output_path.exists() and metadata_path.exists() and not force:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("version") == PREPROCESSING_VERSION:
                return output_path
        except (OSError, json.JSONDecodeError):
            pass

    raw_dir = root / "data" / "raw" / "TJU data"
    payload: dict[str, dict[str, np.ndarray]] = {}
    all_audit: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for name, filename in RAW_FILES.items():
        raw = pd.read_csv(raw_dir / filename, usecols=list(USECOLS))
        rows, cycles = [], []
        for raw_cycle, group in raw.groupby("cycle number", sort=True):
            features = extract_cycle_features(group)
            rows.append(features)
            cycles.append(int(raw_cycle))
        raw_features = np.asarray(rows, dtype=np.float64)
        cycle_values = np.asarray(cycles, dtype=np.int64)
        cleaned, audit = _clean_isolated_3sigma(raw_features, cycle_values)
        for row in audit:
            row.update({"battery": name})
        all_audit.extend(audit)
        payload[name] = {
            "features": cleaned,
            "capacity": cleaned[:, -1].copy(),
            "cycles": cycle_values,
        }
        counts[name] = len(cycle_values)
        print(f"BATTER-MoE private TJU preprocessing {name}: {len(cycle_values)} cycles", flush=True)
        del raw

    private_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_path, payload, allow_pickle=True)
    with audit_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["battery", "cycle", "feature", "raw_value", "mean", "std",
                  "sigma_candidate", "missing", "action", "cleaned_value"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_audit)
    metadata_path.write_text(json.dumps({
        "version": PREPROCESSING_VERSION,
        "scope": "BATTER-MoE private; shared datasets are not modified",
        "sources": {
            "feature_framework": "Wang et al., Nature Communications 15, 4332 (2024)",
            "cleaning": "BATTER-MoE Sec. IV-A",
        },
        "raw_files": RAW_FILES,
        "feature_columns": FEATURE_COLUMNS,
        "cycle_counts": counts,
        "capacity_unit": "Ah",
        "feature_extraction": {
            "charge_curve": f"measured current > {CURRENT_ACTIVITY_THRESHOLD_MA:g} mA",
            "cc_stage": "control/mA > 0 within active charge",
            "cv_stage": "control/V > 0 within active charge",
            "slope": "ordinary least squares against time/s",
            "entropy": f"Shannon entropy of {ENTROPY_BINS}-bin histogram",
            "capacity": "trapezoidal integration of longest contiguous measured-current discharge segment",
        },
        "cleaning": (
            "feature-wise global 3-sigma using sample standard deviation; "
            "interpolate isolated candidates/missing only; retain adjacent runs"
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    print(build(Path(__file__).resolve().parents[2], force=True))
