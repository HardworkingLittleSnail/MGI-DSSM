"""Run and audit the three-dataset efficiency benchmark.

Each model is profiled in an independent Python process so imported author
repositories cannot collide through generic module names such as ``layers``.
Completed JSON files are validated before they are reused.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILER = ROOT / "tools" / "profile_efficiency_rtx3080ti.py"
MODELS = (
    "PatchFormer", "RUL-Mamba", "IC2ML", "BATTER-MoE",
    "Autoformer", "iTransformer", "Ours",
)
DATASETS = ("nasa", "calce", "tju")
EXPECTED_INPUTS = {
    "nasa": {
        "PatchFormer": [1, 16, 1], "RUL-Mamba": [1, 16, 1],
        "IC2ML": [1, 16, 10], "BATTER-MoE": [1, 16, 1],
        "Autoformer": [1, 16, 1], "iTransformer": [1, 16, 4],
        "Ours": [1, 16, 5],
    },
    "calce": {
        "PatchFormer": [1, 64, 1], "RUL-Mamba": [1, 64, 1],
        "IC2ML": [1, 64, 10], "BATTER-MoE": [1, 64, 1],
        "Autoformer": [1, 64, 1], "iTransformer": [1, 64, 4],
        "Ours": [1, 64, 5],
    },
    "tju": {
        "PatchFormer": [1, 64, 1], "RUL-Mamba": [1, 64, 17],
        "IC2ML": [1, 64, 16], "BATTER-MoE": [1, 64, 17],
        "Autoformer": [1, 64, 1], "iTransformer": [1, 64, 4],
        "Ours": [1, 64, 5],
    },
}
TRAIN_BATCHES = {"nasa": 32, "calce": 128, "tju": 128}
EXPECTED_PARAMS = {
    # Independently reproduced from the formal NASA builders and the published
    # BATTER-MoE/PatchFormer source configurations.  CALCE/TJU values are added
    # only where the closed-form GRU count is unambiguous.
    ("nasa", "PatchFormer"): 8567,
    ("nasa", "RUL-Mamba"): 106657,
    ("nasa", "IC2ML"): 1698676,
    ("nasa", "BATTER-MoE"): 94054,
    ("nasa", "Autoformer"): 41985,
    ("nasa", "iTransformer"): 68225,
    ("nasa", "Ours"): 26885,
    ("calce", "Ours"): 12805,
    ("tju", "Ours"): 12773,
}
EXPECTED_ARCHITECTURE = {
    ("nasa", "Ours"): {"hidden_dim": 48, "num_layers": 2, "q_grid_points": 1200},
    ("calce", "Ours"): {"hidden_dim": 32, "num_layers": 2, "q_grid_points": 400},
    ("tju", "Ours"): {"hidden_dim": 48, "num_layers": 1, "q_grid_points": 600},
    ("nasa", "PatchFormer"): {"seq_len": 16, "d_model": 16, "e_layers": 2},
    ("calce", "PatchFormer"): {"seq_len": 64, "d_model": 16, "e_layers": 2},
    ("tju", "PatchFormer"): {"seq_len": 64, "d_model": 16, "e_layers": 2},
    ("nasa", "RUL-Mamba"): {"enc_in": 1, "d_model": 48, "n_dec_layer": 1},
    ("calce", "RUL-Mamba"): {"enc_in": 1, "d_model": 48, "n_dec_layer": 1},
    ("tju", "RUL-Mamba"): {"enc_in": 17, "d_model": 16, "n_dec_layer": 2},
    ("nasa", "IC2ML"): {"context": 16, "input_dim": 10, "hidden_dim": 256},
    ("calce", "IC2ML"): {"context": 64, "input_dim": 10, "hidden_dim": 256},
    ("tju", "IC2ML"): {"context": 64, "input_dim": 16, "hidden_dim": 256},
}


def slug(model: str) -> str:
    return model.lower().replace("-", "_")


def validate(payload: dict, dataset: str, model: str) -> None:
    required = {
        "params", "flops_m", "model_size_mb", "training_time_100_steps_s",
        "inference_median_ms", "inference_iqr_ms",
        "training_peak_memory_mb", "inference_peak_memory_mb",
        "architecture_config", "gpu", "torch",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{dataset}/{model} missing fields: {sorted(missing)}")
    if payload.get("dataset") != dataset or payload.get("model") != model:
        raise ValueError(f"identity mismatch for {dataset}/{model}")
    if payload.get("input_shape_batch1") != EXPECTED_INPUTS[dataset][model]:
        raise ValueError(
            f"input mismatch for {dataset}/{model}: "
            f"{payload.get('input_shape_batch1')} != {EXPECTED_INPUTS[dataset][model]}"
        )
    if int(payload.get("train_batch", -1)) != TRAIN_BATCHES[dataset]:
        raise ValueError(f"train-batch mismatch for {dataset}/{model}")
    if int(payload["params"]) <= 0:
        raise ValueError(f"non-positive parameter count for {dataset}/{model}")
    expected_params = EXPECTED_PARAMS.get((dataset, model))
    if expected_params is not None and int(payload["params"]) != expected_params:
        raise ValueError(
            f"parameter mismatch for {dataset}/{model}: "
            f"{payload['params']} != {expected_params}"
        )
    expected_architecture = EXPECTED_ARCHITECTURE.get((dataset, model), {})
    actual_architecture = payload["architecture_config"]
    for key, expected in expected_architecture.items():
        if actual_architecture.get(key) != expected:
            raise ValueError(
                f"architecture mismatch for {dataset}/{model}/{key}: "
                f"{actual_architecture.get(key)} != {expected}"
            )


def aggregate(output_root: Path, datasets: tuple[str, ...], models: tuple[str, ...]) -> None:
    rows: list[dict] = []
    gpu = torch_version = None
    for dataset in datasets:
        for model in models:
            path = output_root / dataset / f"{slug(model)}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            validate(payload, dataset, model)
            gpu = payload["gpu"] if gpu is None else gpu
            torch_version = payload["torch"] if torch_version is None else torch_version
            if payload["gpu"] != gpu or payload["torch"] != torch_version:
                raise ValueError("all rows must use exactly the same GPU and PyTorch version")
            rows.append(payload)
    (output_root / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fields = [
        "dataset", "model", "params", "params_k", "flops_m", "model_size_mb",
        "training_time_100_steps_s", "inference_median_ms", "inference_iqr_ms",
        "training_peak_memory_mb", "inference_peak_memory_mb", "train_batch",
        "input_shape_batch1", "gpu", "torch", "parameter_basis",
    ]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "outputs" / "efficiency_three_datasets_rtx3080ti",
    )
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--measurement-rounds", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    datasets, models = tuple(args.datasets), tuple(args.models)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        for model in models:
            output = output_root / dataset / f"{slug(model)}.json"
            if output.exists() and not args.force:
                try:
                    validate(json.loads(output.read_text(encoding="utf-8")), dataset, model)
                    print(f"SKIP_VALID {dataset}/{model}", flush=True)
                    continue
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pass
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, str(PROFILER), "--dataset", dataset, "--model", model,
                "--repeats", str(args.repeats), "--warmup", str(args.warmup),
                "--train-steps", str(args.train_steps),
                "--measurement-rounds", str(args.measurement_rounds),
                "--output", str(output),
            ]
            print(f"RUN {dataset}/{model}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            validate(json.loads(output.read_text(encoding="utf-8")), dataset, model)
    aggregate(output_root, datasets, models)
    print(f"COMPLETE {output_root}", flush=True)


if __name__ == "__main__":
    main()
