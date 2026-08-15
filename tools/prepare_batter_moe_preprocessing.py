from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mgi_dssm.preprocessing import capacity_soh, isolated_sigma_interpolate
from mgi_dssm.dataset_paths import processed_dataset_dir


DATASETS = {
    "calce": {
        "directory": "CALCE data",
        "source": "CALCE_Data_raw_unfiltered.npy",
        "output": "CALCE_Data_batter_moe_preprocessed.npy",
        "rated_capacity_ah": 1.1,
    },
    "nasa": {
        "directory": "NASA data",
        "source": "NASA_Data_raw_unfiltered.npy",
        "output": "NASA_Data_batter_moe_preprocessed.npy",
        "rated_capacity_ah": 2.0,
    },
    "tju": {
        "directory": "TJU data",
        "source": "TJU_Data_raw_unfiltered.npy",
        "output": "TJU_Data_batter_moe_preprocessed.npy",
        "rated_capacity_ah": 2.5,
    },
}

IDENTIFIER_COLUMNS = {"BatteryName", "Cycle", "cycle index", "time_idx", "group_id"}


def load_summary(path: Path) -> dict[str, pd.DataFrame]:
    payload = np.load(path, allow_pickle=True)
    obj = payload.item() if payload.shape == () else payload[0]
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


def measurement_columns(frame: pd.DataFrame) -> list[str]:
    selected: list[str] = []
    for column in frame.columns:
        if str(column) in IDENTIFIER_COLUMNS or str(column).endswith("_SOH"):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            selected.append(str(column))
    return selected


def process_dataset(data_dir: Path, dataset: str, window: int, sigma: float) -> None:
    spec = DATASETS[dataset]
    dataset_dir = processed_dataset_dir(data_dir, dataset)
    source_path = dataset_dir / str(spec["source"])
    output_path = dataset_dir / str(spec["output"])
    report_path = dataset_dir / f"{dataset}_batter_moe_preprocessing_audit.csv"
    metadata_path = dataset_dir / f"{dataset}_batter_moe_preprocessing.json"
    if dataset == "calce":
        from mgi_dssm.raw_calce import prepare_calce_raw_dataset

        summary, _ = prepare_calce_raw_dataset(data_dir)
    elif dataset == "nasa":
        from mgi_dssm.raw_nasa import prepare_nasa_raw_dataset

        summary, _ = prepare_nasa_raw_dataset(data_dir)
    elif dataset == "tju":
        from mgi_dssm.raw_tju import prepare_tju_dataset

        summary, _ = prepare_tju_dataset(data_dir)
    else:
        summary = load_summary(source_path)

    cleaned_summary: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, object]] = []
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for battery, source_frame in summary.items():
        if "Capacity" not in source_frame:
            raise ValueError(f"Capacity column missing: {dataset}/{battery}")
        frame = source_frame.copy()
        columns = measurement_columns(frame)
        counts[battery] = {}
        for column in columns:
            raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
            result = isolated_sigma_interpolate(
                raw,
                window=window,
                sigma=sigma,
                min_neighbours=6,
                preserve_endpoints=True,
            )
            frame[column] = result.values
            counts[battery][column] = {
                "sigma_candidates": int(result.sigma_candidate.sum()),
                "isolated_candidates": int(result.isolated.sum()),
                "retained_nonisolated": int((result.sigma_candidate & ~result.isolated).sum()),
                "missing": int(result.missing.sum()),
                "repaired": int(result.repaired.sum()),
            }
            report_indices = np.flatnonzero(result.sigma_candidate | result.missing)
            for index in report_indices:
                if result.isolated[index] and result.repaired[index]:
                    action = "interpolate_isolated_3sigma"
                elif result.missing[index] and result.repaired[index]:
                    action = "interpolate_missing"
                elif result.sigma_candidate[index]:
                    action = "retain_nonisolated_fluctuation"
                else:
                    action = "retain_unbounded_missing"
                audit_rows.append(
                    {
                        "dataset": dataset,
                        "battery": battery,
                        "variable": column,
                        "row_index": int(index),
                        "cycle": frame.iloc[index].get("Cycle", index + 1),
                        "raw_value": raw[index],
                        "local_mean": result.local_mean[index],
                        "local_std": result.local_std[index],
                        "z_score": result.z_score[index],
                        "sigma_candidate": bool(result.sigma_candidate[index]),
                        "isolated": bool(result.isolated[index]),
                        "missing": bool(result.missing[index]),
                        "action": action,
                        "cleaned_value": result.values[index],
                    }
                )

        capacity = pd.to_numeric(frame["Capacity"], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(capacity).all():
            raise ValueError(f"Non-finite capacity remains after cleaning: {dataset}/{battery}")
        frame["Capacity_SOH"] = capacity_soh(capacity, float(spec["rated_capacity_ah"]))
        cleaned_summary[battery] = frame

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_summary(output_path, cleaned_summary)
    with report_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "dataset", "battery", "variable", "row_index", "cycle",
            "raw_value", "local_mean", "local_std", "z_score",
            "sigma_candidate", "isolated", "missing", "action", "cleaned_value",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)
    metadata = {
        "dataset": dataset,
        "source": str(source_path),
        "output": str(output_path),
        "protocol": "BATTER-MoE-compatible published preprocessing",
        "cleaning": {
            "method": "single-pass local isolated sigma detection followed by linear interpolation",
            "window": window,
            "sigma": sigma,
            "sample_standard_deviation_ddof": 1,
            "minimum_neighbours": 6,
            "preserve_endpoints": True,
            "retain_adjacent_candidate_runs": True,
            "rest_period_correction": False,
        },
        "capacity_scaling": {
            "method": "C/C0",
            "rated_capacity_ah": float(spec["rated_capacity_ah"]),
            "stored_column": "Capacity_SOH",
        },
        "other_feature_scaling": (
            "fit feature-wise Min-Max on the training portion of each held-out-cell fold; "
            "reuse for validation/test without clipping"
        ),
        "counts": counts,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{dataset}: saved {output_path}")
    print(f"{dataset}: audit {report_path} rows={len(audit_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create BATTER-MoE-compatible cleaned cycle summaries without overwriting raw data."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--sigma", type=float, default=3.0)
    args = parser.parse_args()
    for dataset in args.datasets:
        process_dataset(args.data_dir, dataset, args.window, args.sigma)


if __name__ == "__main__":
    main()
