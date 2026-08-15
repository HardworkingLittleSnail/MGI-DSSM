"""Train MGI-DSSM on version3 with retained dataset-specific best settings.

The runner evaluates the first cell of each dataset, retains every seed result,
and writes best runs by metric. All physical caches and summaries are resolved
below data/version3 while each dataset keeps its validated preprocessing profile.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from comparison_protocol import DEFAULT_SEEDS, PROTOCOLS
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "our_model_best_configs_version3_10seeds"
DATASETS = ("nasa", "calce", "tju")
SPECS = {
    "nasa": {
        "config": "configs/final_nasa_progressive_best.json",
        "data_dir": "data/version3",
        "summary": "NASA_Data_minimal_interpolated.npy",
        "cache": "physics_curve_cache_nasa_tuned_best.npz",
        "profile": {
            "weak_state_loss_weight": 0.005,
            "thermo_step_scale": 0.03,
            "kinetic_step_scale": 0.05,
            "trend_short_window": 2,
            "trend_long_window": 4,
        },
    },
    "calce": {
        "config": "configs/final_calce_version2_optimized.json",
        "data_dir": "data/version3",
        "summary": "CALCE_Data.npy",
        "cache": "physics_curve_cache_calce_version2_best_profile_v1.npz",
        "profile": {
            "preprocessing_protocol": "legacy",
            "hidden_dim": 32,
            "num_layers": 2,
            "state_supervision": "curve",
            "weak_state_loss_weight": 0.01,
            "lr": 0.0005,
            "threshold_bias_calibration_soh": 0.00012,
            "eol_event_phase_alignment": "none",
        },
    },
    "tju": {
        "config": "configs/final_tju_batter_moe_preprocessed.json",
        "data_dir": "data/version3",
        "summary": "TJU_Data_version2_model_adapter.npy",
        "cache": "physics_curve_cache_tju_version2_best_profile_v1.npz",
        "profile": {"preprocessing_protocol": "batter_moe"},
    },
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def prepare_inputs() -> None:
    version3 = ROOT / "data" / "version3"
    source_path = version3 / "TJU data" / "Dataset_3_NCM_NCA_battery_1C.npy"
    adapter_path = version3 / "TJU data" / "TJU_Data_version2_model_adapter.npy"
    if adapter_path.exists():
        return
    source = np.load(source_path, allow_pickle=True)[0]
    mapping = {"CY25-1": "CY25_1", "CY25-2": "CY25_2", "CY25-3": "CY25_3"}
    adapted = {}
    for name, source_name in mapping.items():
        frame = source[source_name].copy().reset_index(drop=True)
        frame["BatteryName"] = name
        frame["Cycle"] = np.arange(1, len(frame) + 1, dtype=np.int64)
        adapted[name] = frame
    payload = np.empty(1, dtype=object)
    payload[0] = adapted
    np.save(adapter_path, payload, allow_pickle=True)


def result_complete(path: Path, dataset: str, seed: int) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        config = result.get("config", {})
        starts = {int(row["start_point"]) for row in result.get("folds", [])}
        profile = SPECS[dataset]["profile"]
        return (
            config.get("dataset") == dataset
            and int(config.get("seed", -1)) == seed
            and result.get("data_version") == "version3"
            and result.get("processed_summary") == "data/version3"
            and starts == set(PROTOCOLS[dataset].start_points)
            and config.get("summary_filename") == SPECS[dataset]["summary"]
            and config.get("cache_name") == SPECS[dataset]["cache"]
            and all(config.get(key) == value for key, value in profile.items())
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def stream(command: list[str], label: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shown = subprocess.list2cmdline(command)
    print(f"\n{'=' * 20} {label} {'=' * 20}\n{shown}\n", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now().isoformat()}] {shown}\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def train_dataset(dataset: str, seeds: tuple[int, ...], output: Path) -> None:
    spec = SPECS[dataset]
    battery = PROTOCOLS[dataset].batteries[0]
    for seed in seeds:
        run_dir = output / dataset / battery / f"seed_{seed}"
        result_path = run_dir / "results.json"
        if result_complete(result_path, dataset, seed):
            print(f"skip complete: {result_path}", flush=True)
            continue
        command = [
            sys.executable,
            str(ROOT / "run_mgi_dssm.py"),
            "train",
            "--config", str(ROOT / spec["config"]),
            "--data-dir", str(ROOT / spec["data_dir"]),
            "--physics-summary-filename", str(spec["summary"]),
            "--physics-cache-name", str(spec["cache"]),
            "--physics-evaluation-protocol", "patchformer",
            "--test-names", battery,
            "--seed", str(seed),
            "--output-dir", str(run_dir),
        ]
        stream(command, f"our_model/{dataset}/{battery}/seed={seed}", output / "_logs" / dataset / f"seed_{seed}.log")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["data_version"] = "version3"
        result["processed_summary"] = "data/version3"
        write_json(result_path, result)


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(output: Path) -> None:
    rows: list[dict[str, object]] = []
    for path in output.rglob("results.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        config = result.get("config", {})
        dataset = config.get("dataset")
        seed = config.get("seed")
        if dataset not in DATASETS or seed is None:
            continue
        for metric in result.get("folds", []):
            if metric.get("test_battery") != PROTOCOLS[dataset].batteries[0]:
                continue
            rows.append({
                "dataset": dataset,
                "battery": metric["test_battery"],
                "seed": int(seed),
                "start_point": int(metric["start_point"]),
                "MAE": metric.get("mae"),
                "RMSE": metric.get("rmse"),
                "R2": metric.get("r2"),
                "AE": metric.get("AE"),
                "RE": metric.get("RE"),
                "result_file": str(path.relative_to(output)),
            })
    rows.sort(key=lambda row: (str(row["dataset"]), int(row["start_point"]), int(row["seed"])))
    save_csv(output / "all_results.csv", rows)
    write_json(output / "all_results.json", rows)

    best: list[dict[str, object]] = []
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), int(row["start_point"])), []).append(row)
    for (dataset, start), members in groups.items():
        for metric in ("MAE", "RMSE", "R2", "RE"):
            candidates = [row for row in members if row.get(metric) is not None and np.isfinite(float(row[metric]))]
            if not candidates:
                continue
            winner = (max if metric == "R2" else min)(candidates, key=lambda row: float(row[metric]))
            best.append({"selected_metric": metric, **winner})
    save_csv(output / "best_run_by_metric.csv", best)
    write_json(output / "best_run_by_metric.json", best)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    if len(seeds) != 10 or len(set(seeds)) != 10:
        parser.error("formal run requires exactly ten unique seeds")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepare_inputs()
    for dataset in args.datasets:
        train_dataset(dataset, seeds, output)
        aggregate(output)
    print(f"Finished our-model-only training: {output}", flush=True)


if __name__ == "__main__":
    main()
