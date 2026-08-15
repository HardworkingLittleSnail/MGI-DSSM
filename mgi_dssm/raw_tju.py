from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset_paths import processed_dataset_dir, raw_dataset_dir


@dataclass(frozen=True)
class TJUCellSpec:
    filename: str


TJU_CELL_SPECS = {
    "CY25-1": TJUCellSpec("CY25-05_1-#1.csv"),
    "CY25-2": TJUCellSpec("CY25-05_1-#2.csv"),
    "CY25-3": TJUCellSpec("CY25-05_1-#3.csv"),
}

TJU_SUMMARY_NAME = "TJU_Data_raw_unfiltered.npy"
TJU_CURVE_CACHE_NAME = "raw_discharge_curves_batter_moe_v1.npy"


def _primary_discharge_segment(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Return the full, high-voltage discharge when a reference cycle has two."""
    frame = frame.sort_values("time/s").reset_index(drop=True)
    time = frame["time/s"].to_numpy(dtype=np.float64)
    if len(time) < 3:
        return None
    positive_dt = np.diff(time)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0)]
    if not positive_dt.size:
        return None
    gap_limit = max(60.0, 5.0 * float(np.median(positive_dt)))
    boundaries = np.r_[0, np.flatnonzero((np.diff(time) <= 0) | (np.diff(time) > gap_limit)) + 1, len(frame)]
    candidates: list[pd.DataFrame] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        segment = frame.iloc[int(left):int(right)].reset_index(drop=True)
        voltage = segment["Ecell/V"].to_numpy(dtype=np.float64)
        if len(segment) >= 3 and np.isfinite(voltage).all() and np.nanmin(voltage) <= 2.51:
            candidates.append(segment)
    if not candidates:
        return None
    # Reference cycles contain a later auxiliary discharge starting near 3.6 V.
    # The normal capacity trajectory is the segment starting from the highest V.
    return max(candidates, key=lambda item: float(item["Ecell/V"].iloc[0]))


def _integrated_curve(segment: pd.DataFrame, cutoff_voltage_v: float = 2.5) -> dict[str, object] | None:
    time = segment["time/s"].to_numpy(dtype=np.float64)
    voltage = segment["Ecell/V"].to_numpy(dtype=np.float64)
    signed_current = segment["<I>/mA"].to_numpy(dtype=np.float64) / 1000.0
    valid = np.isfinite(time) & np.isfinite(voltage) & np.isfinite(signed_current)
    time, voltage, signed_current = time[valid], voltage[valid], signed_current[valid]
    if len(time) < 3:
        return None
    dt = np.diff(time)
    valid_step = (dt > 0) & (dt <= 60.0)
    q = np.cumsum(np.where(valid_step, dt * (-signed_current[1:]) / 3600.0, 0.0))
    voltage = voltage[1:]
    current = np.abs(signed_current[1:])
    valid_q = np.isfinite(q) & np.isfinite(voltage) & np.isfinite(current) & (q >= 0)
    q, voltage, current = q[valid_q], voltage[valid_q], current[valid_q]
    if len(q) < 2 or q[-1] <= 0:
        return None

    monotone_voltage = np.minimum.accumulate(voltage)
    crossings = np.flatnonzero(monotone_voltage <= cutoff_voltage_v)
    if not crossings.size and monotone_voltage[-1] <= cutoff_voltage_v + 2e-3:
        # The TJU logger terminates some cycles a few 1e-5 V above the nominal
        # 2.5 V threshold. Treat that recorded terminal sample as the cutoff.
        voltage[-1] = cutoff_voltage_v
        return {
            "q": q.astype(np.float64),
            "v": voltage.astype(np.float64),
            "current": float(np.median(current)),
            "r0": 0.05,
        }
    if not crossings.size or crossings[0] == 0:
        return None
    upper = int(crossings[0])
    lower = upper - 1
    v0, v1 = monotone_voltage[lower], monotone_voltage[upper]
    fraction = np.clip((cutoff_voltage_v - v0) / min(v1 - v0, -1e-8), 0.0, 1.0)
    q_cutoff = q[lower] + fraction * (q[upper] - q[lower])
    q = np.r_[q[:upper], q_cutoff]
    voltage = np.r_[voltage[:upper], cutoff_voltage_v]
    current = np.r_[current[:upper], current[upper]]
    return {
        "q": q.astype(np.float64),
        "v": voltage.astype(np.float64),
        "current": float(np.median(current)),
        # TJU has no cycle-wise impedance measurement. This is only the same
        # weak initialization/prior used when another dataset lacks R0.
        "r0": 0.05,
    }


def prepare_tju_dataset(data_dir: Path, force: bool = False) -> tuple[dict[str, pd.DataFrame], dict[str, list[dict[str, object]]]]:
    """Build cycle labels and physical discharge curves directly from TJU raw CSVs."""
    tju_dir = raw_dataset_dir(data_dir, "tju")
    output_dir = processed_dataset_dir(data_dir, "tju")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / TJU_SUMMARY_NAME
    cache_path = output_dir / TJU_CURVE_CACHE_NAME
    if not force and summary_path.exists() and cache_path.exists():
        summary_payload = np.load(summary_path, allow_pickle=True)
        summary = summary_payload[0] if summary_payload.shape else summary_payload.item()
        curves = np.load(cache_path, allow_pickle=True).item()
        if sorted(summary) == sorted(TJU_CELL_SPECS) == sorted(curves):
            return summary, curves

    summary: dict[str, pd.DataFrame] = {}
    curves: dict[str, list[dict[str, object]]] = {}
    usecols = ["time/s", "Ecell/V", "<I>/mA", "cycle number"]
    for name, spec in TJU_CELL_SPECS.items():
        path = tju_dir / spec.filename
        if not path.exists():
            version2_path = tju_dir / "Dataset_3_NCM_NCA_battery" / spec.filename
            if version2_path.exists():
                path = version2_path
        if not path.exists():
            raise FileNotFoundError(f"TJU raw cell not found: {path}")
        raw = pd.read_csv(path, usecols=usecols)
        raw = raw[raw["<I>/mA"] < -1000.0].copy()
        cell_curves: list[dict[str, object]] = []
        raw_cycles: list[int] = []
        available_cycles = np.sort(pd.to_numeric(raw["cycle number"], errors="coerce").dropna().unique())
        skipped: list[int] = []
        for raw_cycle_value in available_cycles:
            raw_cycle = int(raw_cycle_value)
            cycle_frame = raw.loc[raw["cycle number"] == raw_cycle]
            segment = _primary_discharge_segment(cycle_frame)
            curve = None if segment is None else _integrated_curve(segment)
            if curve is None:
                skipped.append(raw_cycle)
                continue
            cell_curves.append(curve)
            raw_cycles.append(raw_cycle)
        capacities = np.asarray([float(row["q"][-1]) for row in cell_curves], dtype=np.float64)
        summary[name] = pd.DataFrame(
            {
                "BatteryName": name,
                "Cycle": np.arange(1, len(cell_curves) + 1, dtype=np.int64),
                "cycle index": np.asarray(raw_cycles, dtype=np.int64),
                "Capacity": capacities,
            }
        )
        curves[name] = cell_curves
        print(
            f"tju_raw battery={name} cycles={len(cell_curves)} "
            f"skipped_invalid={len(skipped)} "
            f"capacity_range={capacities.min():.6f}-{capacities.max():.6f}Ah"
        )
        del raw

    payload = np.empty(1, dtype=object)
    payload[0] = summary
    np.save(summary_path, payload, allow_pickle=True)
    np.save(cache_path, curves, allow_pickle=True)
    return summary, curves
