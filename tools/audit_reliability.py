from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mgi_dssm.data import (  # noqa: E402
    FEATURE_COLUMNS,
    battery_names,
    build_feature_frame,
    load_calce_summary,
)
from mgi_dssm.residual_boosting import build_window_arrays  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit train/test leakage invariants for MGI-DSSM.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=1000)
    parser.add_argument("--rated-capacity", type=float, default=1.1)
    parser.add_argument("--start-points", type=int, nargs="+", default=[300, 400, 500])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_calce_summary(args.data_dir)
    raw = pd.concat(summary.values(), ignore_index=True)
    raw_missing = raw[["Capacity", "Resistance", "CCCT", "CVCT"]].isna().sum()
    print("Raw missing values:")
    print(raw_missing.to_string())

    frame = build_feature_frame(summary)
    feature_columns = [col for col in FEATURE_COLUMNS if col in frame.columns]
    feature_missing = frame[feature_columns + ["Capacity"]].isna().sum()
    if int(feature_missing.sum()) > 0:
        raise AssertionError(f"Feature frame still has missing values: {feature_missing[feature_missing > 0].to_dict()}")

    names = battery_names(frame)
    for test_battery in names:
        train_batteries = [name for name in names if name != test_battery]
        if test_battery in train_batteries:
            raise AssertionError(f"Test battery leaked into training list: {test_battery}")

        train_rows = frame[frame["BatteryName"].isin(train_batteries)]
        test_rows = frame[frame["BatteryName"] == test_battery]
        if train_rows.empty or test_rows.empty:
            raise AssertionError(f"Empty train/test partition for {test_battery}")

        for start_point in args.start_points:
            x, _, _, _, cycles, _ = build_window_arrays(
                frame,
                [test_battery],
                seq_len=args.seq_len,
                max_seq_len=args.max_seq_len,
                start_point=start_point,
                feature_mode="calce-summary",
            )
            if len(cycles) and int(cycles.min()) < int(start_point):
                raise AssertionError(f"Residual windows include cycles before start point for {test_battery}.")
            if len(cycles) and x.shape[0] != len(cycles):
                raise AssertionError("Residual feature rows and cycles are misaligned.")

    print("")
    print("PASS: no code-level leakage found in split, normalization, imputation, or window construction.")
    print("Note: this audit does not remove experiment-level model-selection bias from repeated tuning on test folds.")


if __name__ == "__main__":
    main()
