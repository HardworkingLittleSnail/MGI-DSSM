from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .dataset_paths import processed_dataset_dir, raw_dataset_dir


NASA_BATTERIES = ("B0005", "B0006", "B0007", "B0018")
NASA_CUTOFF_VOLTAGE = {"B0005": 2.7, "B0006": 2.5, "B0007": 2.2, "B0018": 2.5}
NASA_RAW_SUMMARY_NAME = "NASA_Data_raw_unfiltered.npy"
NASA_RAW_CURVE_CACHE_NAME = "raw_discharge_curves_batter_moe_v1.npy"


def _curve_from_record(record: object, cutoff: float, r0: float) -> dict[str, object] | None:
    time = np.asarray(record.data.Time, dtype=np.float64).reshape(-1)
    voltage = np.asarray(record.data.Voltage_measured, dtype=np.float64).reshape(-1)
    current = np.asarray(record.data.Current_measured, dtype=np.float64).reshape(-1)
    size = min(len(time), len(voltage), len(current))
    time, voltage, current = time[:size], voltage[:size], current[:size]
    valid = np.isfinite(time) & np.isfinite(voltage) & np.isfinite(current) & (current < -0.1)
    time, voltage, current = time[valid], voltage[valid], current[valid]
    if len(time) < 3:
        return None
    dt = np.diff(time)
    q = np.cumsum(np.where((dt > 0) & (dt <= 120), -dt * current[1:] / 3600.0, 0.0))
    voltage, current = voltage[1:], np.abs(current[1:])
    keep = np.isfinite(q) & np.isfinite(voltage) & np.isfinite(current) & (q >= 0)
    q, voltage, current = q[keep], voltage[keep], current[keep]
    if len(q) < 2 or q[-1] <= 0:
        return None
    crossing = np.flatnonzero(np.minimum.accumulate(voltage) <= cutoff)
    if crossing.size and crossing[0] > 0:
        upper = int(crossing[0])
        lower = upper - 1
        v0, v1 = voltage[lower], voltage[upper]
        fraction = np.clip((cutoff - v0) / min(v1 - v0, -1e-8), 0.0, 1.0)
        q_cutoff = q[lower] + fraction * (q[upper] - q[lower])
        q = np.r_[q[:upper], q_cutoff]
        voltage = np.r_[voltage[:upper], cutoff]
        current = np.r_[current[:upper], current[upper]]
    elif voltage[-1] <= cutoff + 0.05:
        voltage[-1] = cutoff
    else:
        return None
    return {
        "q": q.astype(np.float64),
        "v": voltage.astype(np.float64),
        "current": float(np.median(current)),
        "r0": float(r0),
    }


def prepare_nasa_raw_dataset(
    data_dir: Path, force: bool = False
) -> tuple[dict[str, pd.DataFrame], dict[str, list[dict[str, object]]]]:
    """Build labels and discharge curves from the official NASA MAT files only."""
    raw_dir = raw_dataset_dir(data_dir, "nasa")
    output_dir = processed_dataset_dir(data_dir, "nasa")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / NASA_RAW_SUMMARY_NAME
    curves_path = output_dir / NASA_RAW_CURVE_CACHE_NAME
    if not force and summary_path.exists() and curves_path.exists():
        payload = np.load(summary_path, allow_pickle=True)
        summary = payload.item() if payload.shape == () else payload[0]
        curves = np.load(curves_path, allow_pickle=True).item()
        if sorted(summary) == sorted(curves) == sorted(NASA_BATTERIES):
            return summary, curves

    summaries: dict[str, pd.DataFrame] = {}
    curves_by_cell: dict[str, list[dict[str, object]]] = {}
    for battery in NASA_BATTERIES:
        path = raw_dir / f"{battery}.mat"
        if not path.exists():
            raise FileNotFoundError(f"NASA official raw cell not found: {path}")
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
        records = np.atleast_1d(payload[battery].cycle)
        latest_r0 = 0.05
        capacities: list[float] = []
        raw_record_indices: list[int] = []
        cell_curves: list[dict[str, object]] = []
        for record_index, record in enumerate(records):
            operation = str(record.type).strip().lower()
            if operation == "impedance":
                resistance = np.asarray(record.data.Re, dtype=np.float64).reshape(-1)
                resistance = resistance[np.isfinite(resistance) & (resistance > 1e-6)]
                if resistance.size:
                    latest_r0 = float(np.median(resistance))
                continue
            if operation != "discharge":
                continue
            official = np.asarray(record.data.Capacity, dtype=np.float64).reshape(-1)
            official = official[np.isfinite(official) & (official > 0)]
            curve = _curve_from_record(record, NASA_CUTOFF_VOLTAGE[battery], latest_r0)
            if not official.size or curve is None:
                continue
            capacities.append(float(official[0]))
            raw_record_indices.append(record_index)
            cell_curves.append(curve)
        if not cell_curves:
            raise ValueError(f"No valid NASA discharge cycles extracted: {battery}")
        summaries[battery] = pd.DataFrame(
            {
                "BatteryName": battery,
                "Cycle": np.arange(1, len(cell_curves) + 1, dtype=np.int64),
                "raw_record_index": raw_record_indices,
                "Capacity": capacities,
            }
        )
        curves_by_cell[battery] = cell_curves
        print(f"nasa_raw battery={battery} cycles={len(cell_curves)}")
    payload = np.empty(1, dtype=object)
    payload[0] = summaries
    np.save(summary_path, payload, allow_pickle=True)
    np.save(curves_path, curves_by_cell, allow_pickle=True)
    return summaries, curves_by_cell
