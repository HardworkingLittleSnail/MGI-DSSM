from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mgi_dssm.physics_train import _mstea_eol_metrics


ROOT = Path("outputs/nasa_sp50_seq16_one_step_10runs")


def main() -> None:
    summary_rows: list[dict[str, object]] = []
    result_paths = sorted(ROOT.glob("run_*_seed_*/results.json"))
    if not result_paths:
        raise FileNotFoundError(f"No NASA results found under {ROOT}")

    for result_path in result_paths:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        predictions = pd.read_csv(result_path.with_name("predictions.csv"))
        threshold = float(payload["config"]["rated_capacity"]) * 0.7
        run_name = result_path.parent.name
        run_index = int(run_name.split("_")[1])
        seed = int(payload["config"]["seed"])

        for fold in payload["folds"]:
            battery = str(fold["test_battery"])
            start_point = int(fold["start_point"])
            selected = predictions[
                (predictions["battery"] == battery)
                & (predictions["start_point"] == start_point)
            ].sort_values("cycle")
            metrics = _mstea_eol_metrics(
                selected["true_ah"].to_numpy(dtype=np.float64),
                selected["pred_ah"].to_numpy(dtype=np.float64),
                threshold,
            )
            fold["RUL_true"] = metrics["rul_true"]
            fold["RUL_pred"] = metrics["rul_pred"]
            fold["AE"] = metrics["ae"]
            fold["RE"] = metrics["re"]
            summary_rows.append(
                {
                    "run": run_index,
                    "seed": seed,
                    "battery": battery,
                    "start_point": start_point,
                    "windows": fold["num_windows"],
                    "mae": fold["mae"],
                    "rmse": fold["rmse"],
                    "r2": fold["r2"],
                    "rul_true": metrics["rul_true"],
                    "rul_pred": metrics["rul_pred"],
                    "ecycle": metrics["ae"],
                    "re": metrics["re"],
                    "persistence_mae": fold["persistence_mae"],
                    "result_path": str(result_path),
                }
            )

        payload["eol_metric"] = {
            "threshold_ah": threshold,
            "AE": "absolute difference between first predicted and true threshold-crossing indices",
            "RE": "AE divided by true RUL from the first evaluated target cycle",
            "censoring": "AE and RE are NaN when the true or predicted trajectory does not cross the threshold",
        }
        result_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    (ROOT / "all_results.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    csv_path = ROOT / "all_results.csv"
    try:
        file = csv_path.open("w", newline="", encoding="utf-8-sig")
    except PermissionError:
        csv_path = ROOT / "all_results_eol_corrected.csv"
        file = csv_path.open("w", newline="", encoding="utf-8-sig")
    with file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Updated {len(result_paths)} runs and {len(summary_rows)} folds")
    print(csv_path)


if __name__ == "__main__":
    main()
