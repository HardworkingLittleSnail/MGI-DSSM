from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial import cKDTree

from mgi_dssm.raw_calce import _first_timestamp, _number, iter_xlsx_rows


FEATURE_NAMES = (
    "qv_Num", "qv_Main_V", "qv_Main_Height", "qv_Area",
    "vq_Num", "vq_Main_V", "vq_Main_Height", "vq_Area",
    "Q_CV_Ah", "Tau_exp", "SE", "MLE",
)

TJU_SOURCE_FEATURES = (
    "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
    "CC Q", "CC charge time", "voltage slope", "voltage entropy",
    "CV Q", "CV charge time", "current slope", "current entropy",
)


def _odd_window(size: int, desired: int = 11) -> int:
    value = min(desired, size if size % 2 else size - 1)
    return max(value, 3)


def _smooth(values: np.ndarray) -> np.ndarray:
    if len(values) < 5:
        return values.astype(np.float64)
    window = _odd_window(len(values))
    order = min(3, window - 1)
    return savgol_filter(values, window_length=window, polyorder=order, mode="interp")


def _charge_arrays(record: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(record.data.Time, dtype=np.float64).reshape(-1)
    voltage = np.asarray(record.data.Voltage_measured, dtype=np.float64).reshape(-1)
    current = np.asarray(record.data.Current_measured, dtype=np.float64).reshape(-1)
    size = min(len(time), len(voltage), len(current))
    time, voltage, current = time[:size], voltage[:size], current[:size]
    valid = np.isfinite(time) & np.isfinite(voltage) & np.isfinite(current) & (current > 0.02)
    time, voltage, current = time[valid], voltage[valid], current[valid]
    if len(time) < 9:
        raise ValueError("charge record has too few valid samples")
    order = np.argsort(time)
    return time[order], voltage[order], current[order]


def _transition_index(voltage: np.ndarray, current: np.ndarray) -> int:
    high = current[current >= np.quantile(current, 0.6)]
    plateau = float(np.median(high)) if len(high) else float(np.max(current))
    candidates = np.flatnonzero((voltage >= 4.0) & (current < 0.95 * plateau))
    if len(candidates):
        return int(np.clip(candidates[0], 3, len(current) - 4))
    return int(np.clip(np.argmax(voltage), 3, len(current) - 4))


def _curve_features(x: np.ndarray, y: np.ndarray, position_is_x: bool) -> tuple[float, ...]:
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique, indices = np.unique(x, return_index=True)
    y = y[indices]
    if len(unique) < 9 or unique[-1] <= unique[0]:
        return (0.0, float(unique[len(unique) // 2]), 0.0, 0.0)
    grid = np.linspace(unique[0], unique[-1], 128)
    sampled = _smooth(np.interp(grid, unique, y))
    derivative = np.gradient(sampled, grid)
    prominence = max(float(np.ptp(derivative)) * 0.05, 1e-9)
    peaks, _ = find_peaks(derivative, prominence=prominence)
    if len(peaks):
        main = int(peaks[np.argmax(derivative[peaks])])
    else:
        main = int(np.argmax(derivative))
    # In Table 3 qv_Main_V is voltage, whereas vq_Main_V is described as
    # capacity position despite the inherited name. In both cases it is the
    # independent-coordinate location of the dominant derivative peak.
    position = float(grid[main])
    return float(len(peaks)), position, float(derivative[main]), float(trapezoid(derivative, grid))


def _cv_features(time: np.ndarray, current: np.ndarray, transition: int) -> tuple[float, float]:
    t = time[transition:] - time[transition]
    i = _smooth(current[transition:])
    q_cv = float(trapezoid(i, t) / 3600.0) if len(t) > 1 else 0.0
    if len(t) < 6 or np.ptp(i) < 1e-6:
        return q_cv, 0.0

    def decay(x, i_inf, amplitude, tau):
        return i_inf + amplitude * np.exp(-x / np.maximum(tau, 1e-8))

    try:
        parameters, _ = curve_fit(
            decay, t, i,
            p0=(max(float(i[-1]), 0.0), max(float(i[0] - i[-1]), 1e-3), max(float(t[-1]) / 3, 1.0)),
            bounds=([0.0, 0.0, 1e-3], [max(float(i.max()), 1.0), 5.0, max(float(t[-1]) * 20, 10.0)]),
            maxfev=5000,
        )
        tau = float(parameters[2])
    except (RuntimeError, ValueError, FloatingPointError):
        tau = 0.0
    return q_cv, tau


def _entropy(values: np.ndarray, bins: int = 32) -> float:
    counts, _ = np.histogram(values, bins=min(bins, max(4, len(values) // 3)))
    probability = counts[counts > 0].astype(np.float64)
    probability /= probability.sum()
    return float(-np.sum(probability * np.log2(probability)))


def _maximum_lyapunov(values: np.ndarray, embedding_dimension: int = 3,
                      delay: int = 2, horizon: int = 8) -> float:
    values = np.asarray(values, dtype=np.float64)
    scale = np.std(values)
    if len(values) < 30 or scale < 1e-10:
        return 0.0
    values = (values - values.mean()) / scale
    count = len(values) - (embedding_dimension - 1) * delay - horizon
    if count < 8:
        return 0.0
    embedded = np.column_stack([
        values[offset:offset + count] for offset in range(0, embedding_dimension * delay, delay)
    ])
    tree = cKDTree(embedded)
    neighbours = np.full(count, -1, dtype=np.int64)
    for index in range(count):
        _, candidates = tree.query(embedded[index], k=min(12, count))
        for candidate in np.atleast_1d(candidates):
            if abs(int(candidate) - index) > horizon:
                neighbours[index] = int(candidate)
                break
    valid = neighbours >= 0
    divergence = []
    for step in range(horizon):
        indices = np.flatnonzero(valid)
        other = neighbours[indices]
        keep = (indices + step < len(values)) & (other + step < len(values))
        distance = np.abs(values[indices[keep] + step] - values[other[keep] + step])
        distance = distance[distance > 1e-8]
        divergence.append(float(np.mean(np.log(distance))) if len(distance) else np.nan)
    divergence = np.asarray(divergence)
    finite = np.isfinite(divergence)
    return float(np.polyfit(np.arange(horizon)[finite], divergence[finite], 1)[0]) if finite.sum() >= 3 else 0.0


def extract_features_from_arrays(time: np.ndarray, voltage: np.ndarray,
                                 current: np.ndarray) -> np.ndarray:
    """Extract the paper's twelve charge-curve features from aligned arrays."""
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    voltage = np.asarray(voltage, dtype=np.float64).reshape(-1)
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    size = min(len(time), len(voltage), len(current))
    time, voltage, current = time[:size], voltage[:size], current[:size]
    valid = np.isfinite(time) & np.isfinite(voltage) & np.isfinite(current) & (current > 0.02)
    time, voltage, current = time[valid], voltage[valid], current[valid]
    if len(time) < 9:
        raise ValueError("charge curve has too few valid samples")
    order = np.argsort(time)
    time, voltage, current = time[order], voltage[order], current[order]
    transition = _transition_index(voltage, current)
    cc_time, cc_voltage, cc_current = time[:transition], voltage[:transition], current[:transition]
    dt = np.diff(cc_time, prepend=cc_time[0])
    dt = np.where((dt > 0) & (dt < 180), dt, 0.0)
    charge = np.cumsum(dt * cc_current / 3600.0)
    voltage_smooth = _smooth(cc_voltage)
    charge_smooth = _smooth(charge)
    qv = _curve_features(voltage_smooth, charge_smooth, True)
    vq = _curve_features(charge_smooth, voltage_smooth, False)
    q_cv, tau = _cv_features(time, current, transition)
    entropy = _entropy(voltage_smooth)
    mle = _maximum_lyapunov(voltage_smooth)
    return np.asarray((*qv, *vq, q_cv, tau, entropy, mle), dtype=np.float64)


def extract_cycle_features(record: object) -> np.ndarray:
    time, voltage, current = _charge_arrays(record)
    return extract_features_from_arrays(time, voltage, current)


def _assemble(summary: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    frame = summary.copy().sort_values("Cycle").reset_index(drop=True)
    features = pd.DataFrame(matrix, columns=FEATURE_NAMES).replace([np.inf, -np.inf], np.nan)
    features = features.ffill().fillna(0.0)
    return pd.concat([
        frame[["BatteryName", "Cycle", "Capacity"]].reset_index(drop=True), features
    ], axis=1)


def extract_nasa_sg_dits_features(raw_directory: Path,
                                  summaries: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Read-only extraction of the paper's F1-F12 feature set from NASA MAT files."""
    output: dict[str, pd.DataFrame] = {}
    for battery, summary in summaries.items():
        records = np.atleast_1d(loadmat(
            raw_directory / f"{battery}.mat", squeeze_me=True, struct_as_record=False
        )[battery].cycle)
        latest: np.ndarray | None = None
        by_discharge: dict[int, np.ndarray] = {}
        for index, record in enumerate(records):
            operation = str(record.type).strip().lower()
            if operation == "charge":
                try:
                    latest = extract_cycle_features(record)
                except (ValueError, np.linalg.LinAlgError):
                    pass
            elif operation == "discharge" and latest is not None:
                by_discharge[index] = latest.copy()
        frame = summary.copy().sort_values("Cycle").reset_index(drop=True)
        rows = [by_discharge.get(int(index)) for index in frame["raw_record_index"]]
        matrix = np.asarray([row if row is not None else np.full(len(FEATURE_NAMES), np.nan)
                             for row in rows], dtype=np.float64)
        # Missing cycles use only earlier same-cell observations. A missing
        # first record is initialized to zero rather than backfilled from future.
        output[battery] = _assemble(frame, matrix)
    return output


def extract_calce_sg_dits_features(raw_directory: Path,
                                   summaries: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Extract F1-F12 from CALCE charge steps without changing the source files."""
    output: dict[str, pd.DataFrame] = {}
    for battery, summary in summaries.items():
        extracted: list[np.ndarray] = []
        discharge_capacities: list[float] = []
        paths = sorted((raw_directory / battery).glob("*.xlsx"), key=_first_timestamp)
        for path in paths:
            groups: dict[int, list[dict[str, str]]] = {}
            for row in iter_xlsx_rows(path):
                try:
                    groups.setdefault(int(float(row["Cycle_Index"])), []).append(row)
                except (KeyError, TypeError, ValueError):
                    continue
            for cycle in sorted(groups):
                cycle_rows = groups[cycle]
                discharge = [row for row in cycle_rows
                             if int(_number(row, "Step_Index", -1)) == 7]
                discharge_time = np.asarray(
                    [_number(row, "Test_Time(s)") for row in discharge], dtype=np.float64
                )
                discharge_current = np.asarray(
                    [_number(row, "Current(A)") for row in discharge], dtype=np.float64
                )
                if len(discharge_time) < 2 or not np.isfinite(discharge_current).any():
                    continue
                discharge_capacities.append(float(-np.sum(
                    np.diff(discharge_time) * discharge_current[1:] / 3600.0
                )))
                rows = [row for row in cycle_rows
                        if int(_number(row, "Step_Index", -1)) in (2, 4)]
                try:
                    extracted.append(extract_features_from_arrays(
                        np.asarray([_number(row, "Test_Time(s)") for row in rows]),
                        np.asarray([_number(row, "Voltage(V)") for row in rows]),
                        np.asarray([_number(row, "Current(A)") for row in rows]),
                    ))
                except (ValueError, np.linalg.LinAlgError):
                    extracted.append(np.full(len(FEATURE_NAMES), np.nan))

        source = np.asarray(extracted, dtype=np.float64)
        if not len(source):
            raise ValueError(f"no CALCE charge cycles extracted for {battery}")
        # Reproduce the documented 40-cycle, two-sigma retention first. The
        # immutable summary was generated by an older integration variant and
        # differs by up to a few rows, so finish alignment on normalized cycle
        # progress without consulting capacity labels.
        kept: list[int] = []
        capacity = np.asarray(discharge_capacities, dtype=np.float64)
        for begin in np.arange(1, len(capacity), 40)[:-1]:
            block = capacity[begin:begin + 40]
            mean, sigma = float(np.mean(block)), float(np.std(block))
            local = np.flatnonzero((block < mean + 2.0 * sigma)
                                   & (block > mean - 2.0 * sigma))
            kept.extend((local + begin).tolist())
        source = source[np.asarray(kept, dtype=np.int64)]
        if len(source) != len(summary):
            source_frame = pd.DataFrame(source, columns=FEATURE_NAMES).ffill().fillna(0.0)
            old_axis = np.linspace(0.0, 1.0, len(source_frame))
            new_axis = np.linspace(0.0, 1.0, len(summary))
            source = np.column_stack([
                np.interp(new_axis, old_axis, source_frame[name].to_numpy(dtype=np.float64))
                for name in FEATURE_NAMES
            ])
        output[battery] = _assemble(summary, source)
    return output


def extract_tju_sg_dits_features(raw_directory: Path,
                                 summaries: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Adapt TJU's released per-cycle charge indicators to 12 SG conditions.

    The version3 TJU files contain per-cycle statistics rather than sampled
    time/voltage/current curves. The diffusion architecture and conditioning
    width remain unchanged; no unavailable curve feature is fabricated.
    """
    output: dict[str, pd.DataFrame] = {}
    for battery, summary in summaries.items():
        ordered = summary.copy().sort_values("Cycle").reset_index(drop=True)
        missing = set(TJU_SOURCE_FEATURES) - set(ordered.columns)
        if missing:
            raise ValueError(f"{battery} lacks TJU charge indicators: {sorted(missing)}")
        matrix = ordered[list(TJU_SOURCE_FEATURES)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
        output[battery] = _assemble(ordered, matrix)
    return output


def extract_sg_dits_features(dataset: str, raw_directory: Path,
                             summaries: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if dataset == "nasa":
        return extract_nasa_sg_dits_features(raw_directory, summaries)
    if dataset == "calce":
        return extract_calce_sg_dits_features(raw_directory, summaries)
    if dataset == "tju":
        return extract_tju_sg_dits_features(raw_directory, summaries)
    raise ValueError(f"unsupported dataset: {dataset}")
