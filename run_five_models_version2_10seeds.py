"""Run seven models on three datasets from the unified version3 root.

Protocol:
  NASA  : B0005, 16 -> 1, SP 50/90
  CALCE : CS2_35, 64 -> 1, SP 200/400
  TJU   : CY25-1, 64 -> 1, SP 200/400

Every task is resumable. Completed result files are retained under one output
root together with per-task logs and final cross-model CSV/JSON summaries.
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

from comparison_protocol import DEFAULT_SEEDS, PROTOCOLS, protocol_manifest
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "seven_models_version3_10seeds"
MODELS = (
    "our_model", "patchformer", "rul-mamba", "batter-moe", "ic2ml",
    "autoformer", "itransformer",
)
DATASETS = ("nasa", "calce", "tju")


def verify_version3() -> None:
    """Fail before training if the unified version3 contract is incomplete."""
    base = ROOT / "data" / "version3"
    required = (
        "manifest.json",
        "NASA data/NASA_Data_minimal_interpolated.npy",
        "NASA data/B0005.mat",
        "NASA data/raw_discharge_curves_batter_moe_v1.npy",
        "CALCE data/CALCE_Data.npy",
        "CALCE data/CS2_35",
        "CALCE data/raw_discharge_curves_batter_moe_v1.npy",
        "TJU data/Dataset_3_NCM_NCA_battery_1C.npy",
        "TJU data/TJU_Data_version2_model_adapter.npy",
        "TJU data/Dataset_3_NCM_NCA_battery",
        "native_inputs/ic2ml/version3/calce_3.6-3.7.npy",
        "native_inputs/ic2ml/version3/nasa_3.9-4.npy",
        "native_inputs/ic2ml/version3/tju_16indicators.npy",
    )
    missing = [item for item in required if not (base / item).exists()]
    if missing:
        raise FileNotFoundError(f"version3 is incomplete; missing: {missing}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def stream(label: str, command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shown = subprocess.list2cmdline(command)
    print(f"\n{'=' * 22} {label} {'=' * 22}\n{shown}\n", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now().isoformat()}] {shown}\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
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


def prepare_our_model_inputs() -> None:
    version2 = ROOT / "data" / "version3"
    source_path = (
        version2 / "TJU data" / "Dataset_3_NCM_NCA_battery_1C.npy"
    )
    adapter_path = version2 / "TJU data" / "TJU_Data_version2_model_adapter.npy"
    if not adapter_path.exists():
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
        print(f"created: {adapter_path}", flush=True)


def own_result_complete(path: Path, seed: int, dataset: str) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        config = result.get("config", {})
        starts = {int(row["start_point"]) for row in result.get("folds", [])}
        return (
            int(config.get("seed", -1)) == seed
            and config.get("dataset") == dataset
            and starts == set(PROTOCOLS[dataset].start_points)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def run_our_model(dataset: str, seeds: tuple[int, ...], output: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "run_our_model_best_configs_10seeds.py"),
        "--datasets", dataset,
        "--seeds", *(str(seed) for seed in seeds),
        "--output-root", str(output / "our_model"),
    ]
    stream(
        f"our_model/{dataset}/best-configs/10seeds",
        command,
        ROOT,
        output / "_logs" / "our_model" / f"{dataset}.log",
    )


def run_native_model(
    model: str, dataset: str, seeds: tuple[int, ...], output: Path, device: str
) -> None:
    data_version = "version3"
    command = [
        sys.executable,
        str(ROOT / "Compare-Models" / "run_native_capacity_models.py"),
        "--model", model,
        "--datasets", dataset,
        "--seeds", *(str(seed) for seed in seeds),
        "--data-version", data_version,
        "--output-root", str(output),
        "--device", device,
    ]
    stream(
        f"{model}/{dataset}/10seeds",
        command,
        ROOT,
        output / "_logs" / model / f"{dataset}.log",
    )


def run_transformer_model(
    model: str, dataset: str, seeds: tuple[int, ...], output: Path, device: str
) -> None:
    data_version = "version3"
    command = [
        sys.executable,
        str(ROOT / "Compare-Models" / "run_autoformer_itransformer.py"),
        "--model", model,
        "--datasets", dataset,
        "--seeds", *(str(seed) for seed in seeds),
        "--data-version", data_version,
        "--output-root", str(output),
        "--device", device,
    ]
    stream(
        f"{model}/{dataset}/10seeds",
        command,
        ROOT,
        output / "_logs" / model / f"{dataset}.log",
    )


def run_batter_moe(dataset: str, seeds: tuple[int, ...], output: Path, device: str) -> None:
    protocol = PROTOCOLS[dataset]
    data_version = "version3"
    common = [
        sys.executable,
        str(ROOT / "Compare-Models" / "MOE" / "run_unified_benchmark.py"),
        "--datasets", dataset,
        "--data-version", data_version,
        "--seeds", *(str(seed) for seed in seeds),
        "--test-batteries", protocol.batteries[0],
        "--output-root", str(output),
        "--device", device,
    ]
    extras = {
        "nasa": [
            "--max-epochs", "100", "--learning-rate", "3e-4",
            "--patience", "20", "--batch-size", "128",
            "--validation-mode", "shuffled", "--capacity-reference", "rated",
        ],
        "calce": [
            "--max-epochs", "100", "--learning-rate", "1e-3",
            "--patience", "10", "--batch-size", "128",
            "--validation-mode", "shuffled", "--capacity-reference", "rated",
        ],
        "tju": [
            "--model-variant", "observation-aware",
            "--max-epochs", "12", "--learning-rate", "1e-4",
            "--patience", "5", "--batch-size", "256",
            "--validation-mode", "shuffled", "--capacity-reference", "initial",
            "--tju-capacity-input-scaling", "c0", "--gradient-clip-norm", "1.0",
            "--lr-plateau-factor", "0.5", "--lr-plateau-patience", "2",
            "--initialize-observation-head-ridge", "--minimum-checkpoint-epoch", "2",
        ],
    }[dataset]
    stream(
        f"batter-moe/{dataset}/10seeds",
        [*common, *extras],
        ROOT,
        output / "_logs" / "batter-moe" / f"{dataset}.log",
    )


def run_ic2ml(dataset: str, seeds: tuple[int, ...], output: Path) -> None:
    protocol = PROTOCOLS[dataset]
    data_version = "version3"
    command = [
        sys.executable,
        "run_rul_benchmark.py",
        "--dataset", dataset,
        "--data-version", data_version,
        "--model-variant", "direct",
        "--test-batteries", protocol.batteries[0],
        "--seeds", *(str(seed) for seed in seeds),
        "--use-capacity-history", "--initialize-history-readout-ridge",
        "--seq-len", str(protocol.seq_len),
        "--epochs", "200", "--batch-size", "128", "--hidden-dim", "256",
        "--learning-rate", "1e-4",
        "--capacity-scaling", "rated",
        "--validation-mode", "chronological", "--selection-objective", "capacity_mae",
        "--patience", "20", "--history-loss-weight", "1",
        "--trajectory-loss-weight", "1", "--rul-loss-weight", "0.5",
        "--capacity-loss", "mse", "--output-root", str(output / "ic2ml"),
    ]
    if dataset == "nasa":
        command += ["--voltage-start", "3.6", "--voltage-end", "3.7"]
    elif dataset == "calce":
        command += ["--voltage-start", "3.6", "--voltage-end", "3.7"]
    elif dataset == "tju":
        command += ["--validation-fraction", "0.15"]
    stream(
        f"ic2ml/{dataset}/10seeds",
        command,
        ROOT / "Compare-Models" / "IC2ML",
        output / "_logs" / "ic2ml" / f"{dataset}.log",
    )


def extract_rows(output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in output.rglob("results.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        relative = path.relative_to(output)
        is_ours = relative.parts and relative.parts[0] == "our_model"
        if not is_ours and result.get("status") != "complete":
            continue
        if is_ours:
            model = "our_model"
            config = result.get("config", {})
            dataset = config.get("dataset")
            seed = config.get("seed")
            metrics = result.get("folds", [])
            battery = metrics[0].get("test_battery") if metrics else None
        else:
            model = result.get("model")
            dataset = result.get("dataset")
            seed = result.get("seed")
            battery = result.get("test_battery")
            metrics = result.get("metrics") or result.get("folds") or []
        if model not in MODELS or dataset not in DATASETS or seed is None:
            continue
        expected_battery = PROTOCOLS[dataset].batteries[0]
        if battery != expected_battery:
            continue
        for metric in metrics:
            start = int(metric["start_point"])
            stage = "early" if start == PROTOCOLS[dataset].start_points[0] else "late"
            rows.append({
                "model": model,
                "dataset": dataset,
                "battery": battery,
                "seed": int(seed),
                "stage": stage,
                "start_point": start,
                "num_windows": metric.get("num_windows"),
                "MAE": metric.get("MAE", metric.get("mae")),
                "RMSE": metric.get("RMSE", metric.get("rmse")),
                "R2": metric.get("R2", metric.get("r2")),
                "RUL_real": metric.get("RUL_real", metric.get("RUL_true")),
                "RUL_pred": metric.get("RUL_pred"),
                "AE": metric.get("AE"),
                "RE": metric.get("RE"),
                "persistence_MAE": metric.get("persistence_mae"),
                "result_file": str(relative),
            })
    rows.sort(key=lambda row: (
        str(row["model"]), str(row["dataset"]), int(row["seed"]), int(row["start_point"])
    ))
    return rows


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    output: Path, requested_seeds: tuple[int, ...],
    requested_models: tuple[str, ...], requested_datasets: tuple[str, ...],
) -> None:
    rows = [
        row for row in extract_rows(output)
        if row["model"] in requested_models and row["dataset"] in requested_datasets
    ]
    save_csv(output / "all_results.csv", rows)
    write_json(output / "all_results.json", rows)

    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["model"], row["dataset"], row["battery"], row["stage"], row["start_point"])
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    for key, members in groups.items():
        base = dict(zip(("model", "dataset", "battery", "stage", "start_point"), key))
        summary: dict[str, object] = {
            **base,
            "runs": len(members),
            "expected_runs": len(requested_seeds),
            "complete": len(members) == len(requested_seeds),
        }
        for metric in ("MAE", "RMSE", "R2", "AE", "RE"):
            values = np.asarray([
                float(row[metric]) for row in members
                if row.get(metric) is not None and np.isfinite(float(row[metric]))
            ])
            summary[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            summary[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) else None
        summaries.append(summary)
        for metric in ("MAE", "RMSE", "R2", "RE"):
            candidates = [
                row for row in members
                if row.get(metric) is not None and np.isfinite(float(row[metric]))
            ]
            if candidates:
                winner = (max if metric == "R2" else min)(
                    candidates, key=lambda row: float(row[metric])
                )
                best_rows.append({"selected_metric": metric, **winner})

    summaries.sort(key=lambda row: (
        str(row["dataset"]), int(row["start_point"]), str(row["model"])
    ))
    best_rows.sort(key=lambda row: (
        str(row["dataset"]), int(row["start_point"]), str(row["model"]),
        str(row["selected_metric"]),
    ))
    save_csv(output / "mean_std_over_10seeds.csv", summaries)
    save_csv(output / "best_run_by_metric.csv", best_rows)
    write_json(output / "mean_std_over_10seeds.json", summaries)
    write_json(output / "best_run_by_metric.json", best_rows)

    expected_groups = len(requested_models) * len(requested_datasets) * 2
    expected_rows = expected_groups * len(requested_seeds)
    completion = {
        "generated_at": datetime.now().isoformat(),
        "result_rows": len(rows),
        "expected_result_rows": expected_rows,
        "complete": len(rows) == expected_rows and len(summaries) == expected_groups,
        "missing_groups": [row for row in summaries if not row["complete"]],
    }
    write_json(output / "completion_report.json", completion)
    print(
        f"aggregate: {len(rows)}/{expected_rows} start-point rows; "
        f"complete={completion['complete']} -> {output}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    verify_version3()

    seeds = tuple(args.seeds)
    models = tuple(args.models)
    datasets = tuple(args.datasets)
    if len(seeds) != 10 or len(set(seeds)) != 10:
        parser.error("formal run requires exactly ten unique seeds")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = protocol_manifest()
    manifest.update({
        "runner": "run_seven_models_version3_10seeds.py",
        "data_sources": {
            "nasa": "data/version3/NASA data/NASA_Data_minimal_interpolated.npy",
            "calce": "data/version3/CALCE data/CALCE_Data.npy",
            "tju": "data/version3/TJU data/Dataset_3_NCM_NCA_battery_1C.npy",
        },
        "models": list(models),
        "datasets": list(datasets),
        "seeds": list(seeds),
        "output_root": str(output),
        "evaluation_scope": "first cell only; rolling one-step; metrics in Ah",
    })
    write_json(output / "protocol.json", manifest)

    if not args.aggregate_only:
        if "our_model" in models:
            prepare_our_model_inputs()
        for model in models:
            for dataset in datasets:
                if model == "our_model":
                    run_our_model(dataset, seeds, output)
                elif model in ("patchformer", "rul-mamba"):
                    run_native_model(model, dataset, seeds, output, args.device)
                elif model in ("autoformer", "itransformer"):
                    run_transformer_model(model, dataset, seeds, output, args.device)
                elif model == "batter-moe":
                    run_batter_moe(dataset, seeds, output, args.device)
                elif model == "ic2ml":
                    run_ic2ml(dataset, seeds, output)
                aggregate(output, seeds, models, datasets)
    aggregate(output, seeds, models, datasets)


if __name__ == "__main__":
    main()
