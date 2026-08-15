from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_FILES = {
    "CY25_1": "CY25-05_1-#1.csv",
    "CY25_2": "CY25-05_1-#2.csv",
    "CY25_3": "CY25-05_1-#3.csv",
}


def _phase_duration(group: pd.DataFrame, mask: np.ndarray) -> float:
    """Integrate a phase duration without counting gaps between segments."""
    time = group["time/s"].to_numpy(dtype=np.float64)
    if time.size < 2:
        return 0.0
    dt = np.diff(time)
    valid = mask[:-1] & mask[1:] & np.isfinite(dt) & (dt >= 0.0) & (dt <= 60.0)
    return float(dt[valid].sum())


def _extract_cycles(path: Path, battery: str) -> pd.DataFrame:
    columns = [
        "time/s",
        "control/V/mA",
        "Ecell/V",
        "<I>/mA",
        "Q discharge/mA.h",
        "Q charge/mA.h",
        "control/V",
        "control/mA",
        "cycle number",
    ]
    raw = pd.read_csv(path, usecols=columns)
    rows: list[dict[str, float | int | str]] = []
    for cycle, group in raw.groupby("cycle number", sort=True):
        control_v = group["control/V"].to_numpy(dtype=np.float64)
        control_ma = group["control/mA"].to_numpy(dtype=np.float64)
        combined = group["control/V/mA"].to_numpy(dtype=np.float64)
        cc_mask = (control_ma > 1000.0) & (control_v < 1.0)
        cv_mask = control_v > 4.0
        discharge_mask = (control_ma < -2000.0) | (combined < -2000.0)
        rows.append(
            {
                "BatteryName": battery,
                "Cycle": int(cycle),
                "Capacity_raw": float(group["Q discharge/mA.h"].max()) / 1000.0,
                "CC charge time_raw": _phase_duration(group, cc_mask),
                "CV charge time_raw": _phase_duration(group, cv_mask),
                "CC discharge time_raw": _phase_duration(group, discharge_mask),
            }
        )
    return pd.DataFrame(rows)


def _local_three_sigma_repair(values: pd.Series, radius: int = 10) -> tuple[pd.Series, np.ndarray]:
    """Repair isolated dead points using neighbour-only local mean ± 3 sigma."""
    source = pd.to_numeric(values, errors="coerce").astype(float)
    repaired = source.copy()
    flagged = np.zeros(len(source), dtype=bool)
    # Two passes catch the paired spike/recovery pattern around characterization cycles.
    for _ in range(2):
        arr = repaired.to_numpy(dtype=np.float64)
        current_flags = np.zeros(len(arr), dtype=bool)
        for index, value in enumerate(arr):
            lo = max(0, index - radius)
            hi = min(len(arr), index + radius + 1)
            neighbours = np.concatenate((arr[lo:index], arr[index + 1 : hi]))
            neighbours = neighbours[np.isfinite(neighbours)]
            if neighbours.size < 6 or not np.isfinite(value):
                current_flags[index] = not np.isfinite(value)
                continue
            mean = float(neighbours.mean())
            std = float(neighbours.std(ddof=1))
            threshold = max(3.0 * std, 1e-10)
            current_flags[index] = abs(value - mean) > threshold
        flagged |= current_flags
        candidate = repaired.mask(current_flags)
        repaired = candidate.interpolate(method="linear", limit_direction="both")
    return repaired, flagged


def prepare(raw_dir: Path, output: Path, radius: int) -> None:
    batteries: dict[str, pd.DataFrame] = {}
    for battery, filename in TARGET_FILES.items():
        frame = _extract_cycles(raw_dir / filename, battery)
        repair_counts: dict[str, int] = {}
        for name in ["Capacity", "CC charge time", "CV charge time", "CC discharge time"]:
            repaired, flags = _local_three_sigma_repair(frame[f"{name}_raw"], radius=radius)
            frame[name] = repaired
            repair_counts[name] = int(flags.sum())
        # The common loader expects this column. TJU has no per-cycle resistance;
        # load_cycle_summary maps CC discharge time to the third physical indicator.
        frame["Resistance"] = np.nan
        batteries[battery] = frame
        print(
            f"{battery}: cycles={len(frame)} range={frame.Cycle.min()}-{frame.Cycle.max()} "
            f"repairs={repair_counts} capacity=[{frame.Capacity.iloc[0]:.6f},"
            f"{frame.Capacity.iloc[-1]:.6f}]Ah"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.array([batteries], dtype=object), allow_pickle=True)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=10)
    args = parser.parse_args()
    prepare(args.raw_dir, args.output, args.radius)


if __name__ == "__main__":
    main()
