"""Replace the IC2ML rows in the seven-model aggregate with aligned results."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "seven_models_version3_10seeds"
SOURCE_DIR = ROOT / "outputs" / "ic2ml_aligned_version3_10seeds"
CSV_PATH = TABLE_DIR / "all_results.csv"
JSON_PATH = TABLE_DIR / "all_results.json"
FALLBACK_CSV_PATH = TABLE_DIR / "all_results_ic2ml_updated.csv"
FIELDS = [
    "model", "dataset", "battery", "seed", "stage", "start_point",
    "num_windows", "MAE", "RMSE", "R2", "RUL_real", "RUL_pred",
    "AE", "RE", "persistence_MAE", "result_file",
]


def main() -> None:
    old_rows = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    retained = [row for row in old_rows if row.get("model") != "ic2ml"]

    new_rows: list[dict[str, object]] = []
    result_paths = sorted(SOURCE_DIR.rglob("results.json"))
    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        folds = payload["folds"]
        early_start = min(int(fold["start_point"]) for fold in folds)
        for fold in folds:
            start = int(fold["start_point"])
            new_rows.append({
                "model": "ic2ml",
                "dataset": payload["dataset"],
                "battery": payload["test_battery"],
                "seed": int(payload["seed"]),
                "stage": "early" if start == early_start else "late",
                "start_point": start,
                "num_windows": int(fold["num_windows"]),
                "MAE": fold["mae"],
                "RMSE": fold["rmse"],
                "R2": fold["r2"],
                "RUL_real": fold.get("RUL_real"),
                "RUL_pred": fold.get("RUL_pred"),
                "AE": fold.get("AE"),
                "RE": fold.get("RE"),
                "persistence_MAE": fold.get("persistence_mae"),
                "result_file": path.relative_to(TABLE_DIR.parent).as_posix(),
            })

    expected = {"nasa": 20, "calce": 20, "tju": 20}
    counts = {
        dataset: sum(row["dataset"] == dataset for row in new_rows)
        for dataset in expected
    }
    if len(result_paths) != 30 or counts != expected:
        raise RuntimeError(
            f"incomplete IC2ML results: files={len(result_paths)}, rows={counts}"
        )

    rows = retained + new_rows
    rows.sort(key=lambda row: (
        str(row["model"]), str(row["dataset"]), str(row["battery"]),
        int(row["seed"]), int(row["start_point"]),
    ))
    JSON_PATH.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    try:
        csv_stream = CSV_PATH.open("w", newline="", encoding="utf-8")
        written_csv = CSV_PATH
    except PermissionError:
        csv_stream = FALLBACK_CSV_PATH.open("w", newline="", encoding="utf-8")
        written_csv = FALLBACK_CSV_PATH
    with csv_stream as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in FIELDS} for row in rows)

    print(
        f"total_rows={len(rows)} retained_rows={len(retained)} "
        f"ic2ml_rows={len(new_rows)} csv={written_csv.name}"
    )
    for dataset in ("nasa", "calce", "tju"):
        selected = [row for row in new_rows if row["dataset"] == dataset]
        mean_mae = sum(float(row["MAE"]) for row in selected) / len(selected)
        print(f"{dataset}: rows={len(selected)} mean_MAE={mean_mae:.9f}")


if __name__ == "__main__":
    main()
