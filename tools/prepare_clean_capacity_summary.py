from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def local_sigma_interpolate(
    values: np.ndarray,
    window: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace only local mean +/- sigma outliers by linear interpolation."""
    y = np.asarray(values, dtype=np.float64).copy()
    flagged = np.zeros(len(y), dtype=bool)
    if window < 5:
        return y, flagged
    radius = window // 2
    for index, value in enumerate(y):
        left, right = max(0, index - radius), min(len(y), index + radius + 1)
        neighbours = np.r_[y[left:index], y[index + 1:right]]
        neighbours = neighbours[np.isfinite(neighbours)]
        if neighbours.size >= 4:
            mean, std = float(neighbours.mean()), float(neighbours.std())
            if std > 0.0 and abs(value - mean) > sigma * std:
                flagged[index] = True
    valid = np.flatnonzero(~flagged & np.isfinite(y))
    if flagged.any() and valid.size >= 2:
        y[flagged] = np.interp(np.flatnonzero(flagged), valid, y[valid])
    return y, flagged


def load_summary(path: Path) -> dict[str, pd.DataFrame]:
    obj = np.load(path, allow_pickle=True)
    obj = obj.item() if obj.shape == () else obj[0]
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected summary structure: {path}")
    return {
        str(name): frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
        for name, frame in obj.items()
    }


def save_summary(path: Path, summary: dict[str, pd.DataFrame]) -> None:
    payload = np.empty(1, dtype=object)
    payload[0] = summary
    np.save(path, payload, allow_pickle=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a locally cleaned capacity summary without overwriting its source."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--sigma", type=float, default=3.0)
    args = parser.parse_args()

    summary = load_summary(args.input)
    cleaned: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for battery, frame in summary.items():
        out = frame.copy()
        capacity = pd.to_numeric(out["Capacity"], errors="coerce").to_numpy(dtype=np.float64)
        clean_capacity, flagged = local_sigma_interpolate(capacity, args.window, args.sigma)
        out["Capacity"] = clean_capacity
        cleaned[battery] = out
        for index in np.flatnonzero(flagged):
            rows.append(
                {
                    "battery": battery,
                    "cycle": int(out.iloc[index]["Cycle"]),
                    "raw_capacity": float(capacity[index]),
                    "clean_capacity": float(clean_capacity[index]),
                    "delta_ah": float(clean_capacity[index] - capacity[index]),
                    "window": int(args.window),
                    "sigma": float(args.sigma),
                }
            )
        print(
            f"clean_capacity battery={battery} window={args.window} "
            f"sigma={args.sigma:g} flagged={int(flagged.sum())}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_summary(args.output, cleaned)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "battery", "cycle", "raw_capacity", "clean_capacity",
            "delta_ah", "window", "sigma",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_clean_summary={args.output}")
    print(f"saved_clean_report={args.report}")


if __name__ == "__main__":
    main()
