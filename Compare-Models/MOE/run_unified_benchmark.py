"""BATTER-MoE on the shared three-dataset one-step protocol."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from scipy.io import loadmat


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from comparison_protocol import (  # noqa: E402
    DEFAULT_SEEDS, PROTOCOLS, evaluate_predictions,
    load_summary, protocol_manifest, seed_everything, write_csv, write_json,
)
from batter_moe.config import ExperimentConfig, ModelConfig, get_paper_config  # noqa: E402
from batter_moe.data import CellSeries, MinMaxScaler, WindowDataset  # noqa: E402
from batter_moe.model import BATTERMoE  # noqa: E402
from batter_moe.train import fit, predict  # noqa: E402
from prepare_tju_native_features import build as build_tju_features  # noqa: E402

RUNNER_VERSION = "paper-absolute-onestep-v16-ridge-initialized-readout"


class LastCapacityWarmupDataset(torch.utils.data.Dataset):
    """Initialize the capacity path without relying on cell-specific covariates."""

    def __init__(self, source):
        self.source = source

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        window, _ = self.source[index]
        capacity_only = torch.zeros_like(window)
        capacity_only[:, 16] = window[:, 16]
        return capacity_only, window[-1, 16]


def build_config(
    dataset: str, battery: str, max_epochs: int | None, model_variant: str = "full"
) -> ExperimentConfig:
    protocol = PROTOCOLS[dataset]
    train_cells = tuple(name for name in protocol.batteries if name != battery)
    if dataset in ("nasa", "tju"):
        base = get_paper_config(dataset)
        config = replace(base, train_cells=train_cells, test_cell=battery, start_points=protocol.start_points,
                         eol_fraction=protocol.eol_fraction)
    else:
        # CALCE is not in the BATTER-MoE paper. Use its compact univariate NASA
        # backbone and adapt only the common 64-cycle window/patch scales.
        model = ModelConfig(1, 64, (8, 16, 32), 64, 1, 128, 4, dropout=0.05)
        config = ExperimentConfig(dataset, protocol.rated_capacity, protocol.eol_fraction,
                                  train_cells, battery, protocol.start_points, model)
    if max_epochs is not None:
        config = replace(config, max_epochs=max_epochs, patience=min(config.patience, max_epochs))
    if model_variant == "no-ct":
        config = replace(config, model=replace(config.model, use_cross_time=False))
    elif model_variant == "observation-aware":
        config = replace(
            config,
            model=replace(config.model, use_latest_observation_readout=True),
        )
    return config


def load_cells(
    dataset: str, tju_input_source: str = "private",
    data_version: str = "processed",
) -> dict[str, CellSeries]:
    protocol = PROTOCOLS[dataset]
    if data_version in ("version2.0", "version3"):
        base = ROOT / "data" / ("version3" if data_version == "version3" else "processed-version2.0")
        if dataset == "nasa":
            if data_version == "version3":
                payload = np.load(
                    base / "NASA data" / "NASA_Data_minimal_interpolated.npy",
                    allow_pickle=True,
                )[0]
                return {
                    name: CellSeries(
                        name,
                        payload[name]["Capacity"].to_numpy(dtype=np.float64)[:, None],
                        payload[name]["Capacity"].to_numpy(dtype=np.float64),
                        payload[name]["Cycle"].to_numpy(dtype=np.int64),
                    )
                    for name in protocol.batteries
                }
            result = {}
            for name in protocol.batteries:
                structure = loadmat(
                    base / "NASA data" / f"{name}.mat",
                    squeeze_me=True, struct_as_record=False,
                )[name]
                capacity = np.asarray([
                    float(cycle.data.Capacity)
                    for cycle in np.atleast_1d(structure.cycle)
                    if cycle.type == "discharge"
                ], dtype=np.float64)
                cycles = np.arange(1, len(capacity) + 1, dtype=np.int64)
                result[name] = CellSeries(name, capacity[:, None], capacity, cycles)
            return result
        if dataset == "tju":
            payload = np.load(
                base / "TJU data" / "Dataset_3_NCM_NCA_battery_1C.npy",
                allow_pickle=True,
            )[0]
            mapping = {"CY25-1": "CY25_1", "CY25-2": "CY25_2", "CY25-3": "CY25_3"}
            indicators = [
                "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
                "CC Q", "CC charge time", "voltage slope", "voltage entropy",
                "current mean", "current std", "current kurtosis", "current skewness",
                "CV Q", "CV charge time", "current slope", "current entropy",
            ]
            result = {}
            for name, source_name in mapping.items():
                frame = payload[source_name].sort_values("Cycle").reset_index(drop=True)
                capacity = frame["Capacity"].to_numpy(dtype=np.float64)
                features = np.column_stack((
                    frame[indicators].to_numpy(dtype=np.float64), capacity
                ))
                result[name] = CellSeries(
                    name, features, capacity,
                    frame["Cycle"].to_numpy(dtype=np.int64),
                )
            return result
        if dataset == "calce":
            payload = np.load(
                base / "CALCE data" / "CALCE_Data.npy", allow_pickle=True
            )[0]
            return {
                name: CellSeries(
                    name,
                    payload[name]["Capacity"].to_numpy(dtype=np.float64)[:, None],
                    payload[name]["Capacity"].to_numpy(dtype=np.float64),
                    payload[name]["Cycle"].to_numpy(dtype=np.int64),
                )
                for name in protocol.batteries
            }
        raise ValueError(f"{data_version} is not configured for {dataset}")
    if dataset != "tju":
        frames = load_summary(protocol)
        return {
            name: CellSeries(name, frame["Capacity"].to_numpy(dtype=np.float64)[:, None],
                             frame["Capacity"].to_numpy(dtype=np.float64),
                             frame["Cycle"].to_numpy(dtype=np.int64))
            for name, frame in frames.items()
        }
    if tju_input_source == "aligned-reference":
        path = ROOT / "data" / "processed" / "TJU data" / "batter_moe_17_features_aligned.npy"
        payload = np.load(path, allow_pickle=True).item()
        return {
            name: CellSeries(
                name,
                np.asarray(payload[name]["features"], dtype=np.float64),
                np.asarray(payload[name]["features"], dtype=np.float64)[:, -1],
                np.asarray(payload[name]["cycles"], dtype=np.int64),
            )
            for name in protocol.batteries
        }
    path = build_tju_features(ROOT)
    payload = np.load(path, allow_pickle=True).item()
    return {
        name: CellSeries(name, np.asarray(payload[name]["features"], dtype=np.float64),
                         np.asarray(payload[name]["capacity"], dtype=np.float64),
                         np.asarray(payload[name]["cycles"], dtype=np.int64))
        for name in protocol.batteries
    }


def prepare(
    config: ExperimentConfig, cells: dict[str, CellSeries],
    validation_mode: str = "shuffled", capacity_reference: str = "rated",
    tju_capacity_input_scaling: str = "minmax",
    window_normalize_noncapacity: bool = False,
):
    # BATTER-MoE uses a fixed shuffled 80/20 split over pooled training
    # windows.  Keep that native selection protocol while the held-out cell
    # and the shared cleaned capacity labels remain fixed by our benchmark.
    if validation_mode == "cell":
        fit_names = config.train_cells[:-1]
        validation_names = config.train_cells[-1:]
    else:
        fit_names = config.train_cells
        validation_names = ()
    raw_pooled = WindowDataset(
        [cells[name] for name in fit_names], config.model.lookback
    )
    generator = np.random.default_rng(config.split_seed)
    indices = generator.permutation(len(raw_pooled))
    split = (
        len(indices) if validation_mode == "cell"
        else int(round(len(indices) * (1.0 - config.validation_fraction)))
    )
    train_indices = indices[:split]
    validation_indices = indices[split:]
    scaler = None
    normalized: dict[str, CellSeries] = {}
    if config.dataset == "tju":
        feature_count = 17 if tju_capacity_input_scaling == "minmax" else 16
        train_rows = raw_pooled.windows[train_indices, :, :feature_count].numpy().reshape(
            -1, feature_count
        )
        scaler = MinMaxScaler.fit([train_rows])
    for name, cell in cells.items():
        capacity_scale = (
            float(cell.capacity[0]) if capacity_reference == "initial"
            else config.rated_capacity
        )
        if config.dataset == "tju":
            features = cell.features.copy()
            if tju_capacity_input_scaling == "minmax":
                # Min-Max scaling of raw capacity is algebraically identical
                # to first applying C/C0 and then Min-Max scaling.
                features = scaler.transform(features)
            else:
                features[:, :16] = scaler.transform(features[:, :16])
                features[:, 16] = cell.capacity / capacity_scale
        else:
            features = cell.capacity[:, None] / capacity_scale
        normalized[name] = CellSeries(
            name, features, cell.capacity / capacity_scale, cell.cycles
        )
    pooled = WindowDataset([normalized[name] for name in fit_names], config.model.lookback)
    train = torch.utils.data.Subset(pooled, train_indices.tolist())
    validation = (
        WindowDataset([normalized[name] for name in validation_names], config.model.lookback)
        if validation_mode == "cell"
        else torch.utils.data.Subset(pooled, validation_indices.tolist())
    )
    test = WindowDataset([normalized[config.test_cell]], config.model.lookback)
    if config.dataset == "tju" and window_normalize_noncapacity:
        datasets = [pooled, test]
        if validation_mode == "cell":
            datasets.append(validation)
        for window_dataset in datasets:
            values = window_dataset.windows[:, :, :16]
            mean = values.mean(dim=1, keepdim=True)
            std = values.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            window_dataset.windows[:, :, :16] = (values - mean) / std
    return train, validation, test, scaler


def run_fold(
    dataset, battery, seed, output_root, device, max_epochs=None, force=False,
    tju_input_source="private", model_variant="full", data_version="processed",
    learning_rate=None, patience=None, batch_size=None, validation_mode="shuffled",
    capacity_reference="rated", gradient_clip_norm=None,
    lr_plateau_factor=None, lr_plateau_patience=None,
    initialize_head_to_target_median=False,
    tju_capacity_input_scaling="minmax",
    capacity_warmup_epochs=0,
    capacity_prior_tokenizer_init=False,
    window_normalize_noncapacity=False,
    initialize_observation_head_ridge=False,
    minimum_checkpoint_epoch=0,
):
    protocol = PROTOCOLS[dataset]
    config = build_config(dataset, battery, max_epochs, model_variant)
    config = replace(
        config,
        learning_rate=(config.learning_rate if learning_rate is None else learning_rate),
        patience=(config.patience if patience is None else patience),
        batch_size=(config.batch_size if batch_size is None else batch_size),
        gradient_clip_norm=(config.gradient_clip_norm if gradient_clip_norm is None
                            else gradient_clip_norm),
        lr_plateau_factor=(config.lr_plateau_factor if lr_plateau_factor is None
                           else lr_plateau_factor),
        lr_plateau_patience=(config.lr_plateau_patience if lr_plateau_patience is None
                             else lr_plateau_patience),
    )
    implementation_version = RUNNER_VERSION + "-absolute-readout" + (
        "-aligned-reference" if tju_input_source == "aligned-reference" else "-private-input"
    ) + (
        f"-{model_variant}-{data_version}-lr{config.learning_rate:g}"
        f"-bs{config.batch_size}-ep{config.max_epochs}-pat{config.patience}"
        f"-clip{config.gradient_clip_norm:g}-plateau{config.lr_plateau_factor:g}x"
        f"{config.lr_plateau_patience}-headmedian{int(initialize_head_to_target_median)}"
        f"-warm{capacity_warmup_epochs}-capprior{int(capacity_prior_tokenizer_init)}"
        f"-capinput{tju_capacity_input_scaling}"
        f"-winnorm{int(window_normalize_noncapacity)}-val{validation_mode}"
        f"-ridgeinit{int(initialize_observation_head_ridge)}"
        f"-minepoch{minimum_checkpoint_epoch}-c0{capacity_reference}"
    )
    if dataset == "nasa" and data_version == "processed":
        implementation_version += "-nasa-minimal-interpolated-v1"
    output = output_root / "batter-moe" / dataset / battery / f"seed_{seed}"
    result_path = output / "results.json"
    if result_path.exists() and not force:
        try:
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if (cached.get("status") == "complete"
                    and cached.get("implementation_version") == implementation_version):
                print(f"skip complete: {output}", flush=True)
                return
        except (OSError, json.JSONDecodeError):
            pass
    output.mkdir(parents=True, exist_ok=True)
    write_json(result_path, {
        "status": "running", "model": "batter-moe", "dataset": dataset,
        "test_battery": battery, "seed": seed,
        "implementation_version": implementation_version,
        "readout_variant": "paper-absolute",
        "model_variant": model_variant,
    })
    cells = load_cells(dataset, tju_input_source, data_version)
    train, validation, test, scaler = prepare(
        config, cells, validation_mode, capacity_reference,
        tju_capacity_input_scaling,
        window_normalize_noncapacity,
    )
    seed_everything(seed)
    model = BATTERMoE(config.model)
    if capacity_prior_tokenizer_init:
        if dataset != "tju":
            raise ValueError("Capacity-prior tokenizer initialization is TJU-only.")
        with torch.no_grad():
            for patch, projection in zip(
                config.model.patch_lengths, model.tokenizer.projections
            ):
                projection.weight.zero_()
                projection.bias.zero_()
                basis = torch.empty(
                    config.model.d_model,
                    device=projection.weight.device,
                    dtype=projection.weight.dtype,
                ).normal_(mean=0.0, std=0.02)
                for time_index in range(patch):
                    projection.weight[:, time_index * 17 + 16] = basis / patch
    if initialize_observation_head_ridge:
        if not config.model.use_latest_observation_readout:
            raise ValueError("Ridge head initialization requires observation-aware readout.")
        if not isinstance(train, torch.utils.data.Subset):
            raise ValueError("Ridge head initialization expects the shuffled training subset.")
        train_windows = train.dataset.windows[train.indices, -1, :].numpy()
        train_targets = train.dataset.targets[train.indices].numpy()
        ridge = Ridge(alpha=1e-6).fit(train_windows, train_targets)
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.weight[0, config.model.d_model:] = torch.as_tensor(
                ridge.coef_, dtype=model.head.weight.dtype
            )
            model.head.bias.fill_(float(ridge.intercept_))
    elif initialize_head_to_target_median:
        # Training-only initialization for stable absolute regression.  This
        # does not add a persistence/residual path and does not use test-cell
        # information; the paper's linear head remains the sole readout.
        if isinstance(train, torch.utils.data.Subset):
            train_targets = train.dataset.targets[train.indices]
        else:
            train_targets = train.targets
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.fill_(float(train_targets.median()))
    started = time.time()
    warmup_history = []
    if capacity_warmup_epochs:
        if dataset != "tju" or tju_capacity_input_scaling != "c0":
            raise ValueError("Capacity warmup requires TJU with C/C0 capacity input.")
        warmup_config = replace(
            config,
            max_epochs=capacity_warmup_epochs,
            patience=min(5, capacity_warmup_epochs),
        )
        model, warmup_history = fit(
            model,
            LastCapacityWarmupDataset(train),
            LastCapacityWarmupDataset(validation),
            warmup_config,
            device,
            seed,
            verbose_prefix=f"[batter-moe/{dataset}/{battery}/seed={seed}/warmup]",
        )
    model, history = fit(
        model, train, validation, config, device, seed,
        verbose_prefix=f"[batter-moe/{dataset}/{battery}/seed={seed}]",
        minimum_checkpoint_epoch=minimum_checkpoint_epoch,
    )
    pred_norm, true_norm = predict(model, test, config.batch_size, device)
    capacity_scale = (
        float(cells[battery].capacity[0])
        if capacity_reference == "initial" else protocol.rated_capacity
    )
    y_true, y_pred = true_norm * capacity_scale, pred_norm * capacity_scale
    if cells[battery].cycles is None:
        raise ValueError(f"Missing cycle numbers for {dataset}/{battery}")
    cycles = np.asarray(cells[battery].cycles, dtype=int)[protocol.seq_len:]
    metrics, rows = evaluate_predictions(cycles, y_true, y_pred, protocol)
    for row in rows:
        row.update({"model": "batter-moe", "dataset": dataset, "battery": battery, "seed": seed})
    result = {
        "status": "complete", "model": "batter-moe", "dataset": dataset,
        "implementation_version": implementation_version,
        "readout_variant": "paper-absolute",
        "model_variant": model_variant,
        "test_battery": battery, "train_batteries": list(config.train_cells), "seed": seed,
        "task": f"{protocol.seq_len}-cycle rolling one-step capacity prediction",
        "start_points": list(protocol.start_points), "metrics": metrics,
        "best_epoch": int(min(
            (row for row in history if row["epoch"] >= minimum_checkpoint_epoch),
            key=lambda row: row["validation_mae"],
        )["epoch"]),
        "elapsed_seconds": time.time() - started, "paper_config": config.to_dict(),
        "native_input": "17 cycle-level indicators" if dataset == "tju" else "univariate capacity",
        "data_version": data_version,
        "validation_mode": validation_mode,
        "capacity_reference": capacity_reference,
        "capacity_warmup_epochs": capacity_warmup_epochs,
        "observation_head_ridge_initialization": initialize_observation_head_ridge,
        "tju_input_source": tju_input_source if dataset == "tju" else None,
        "processed_summary": (
            f"data/{'version3' if data_version == 'version3' else 'processed-version2.0'}"
            if data_version in ("version2.0", "version3") else (
            (
                "data/processed/TJU data/batter_moe_17_features_aligned.npy"
                if tju_input_source == "aligned-reference"
                else "Compare-Models/MOE/private_data/tju_17_features_batter_moe.npy"
            )
            if dataset == "tju" else str(protocol.summary_path.relative_to(ROOT))
        )),
        "feature_scaler": None if scaler is None else {"minimum": scaler.minimum.tolist(), "maximum": scaler.maximum.tolist()},
    }
    torch.save({"state_dict": model.state_dict(), "result": result}, output / "checkpoint.pt")
    write_csv(output / "training_history.csv", history)
    if warmup_history:
        write_csv(output / "warmup_history.csv", warmup_history)
    write_csv(output / "predictions.csv", rows)
    write_json(result_path, result)
    print(f"complete: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=tuple(PROTOCOLS), default=list(PROTOCOLS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--test-batteries", nargs="+", default=None)
    parser.add_argument(
        "--data-version", choices=("processed", "version2.0", "version3"), default="version3"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "comparison_models_native_10seeds")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--lr-plateau-factor", type=float, default=None)
    parser.add_argument("--lr-plateau-patience", type=int, default=None)
    parser.add_argument(
        "--initialize-head-to-target-median", action="store_true",
        help="Initialize the unchanged linear head from training targets only.",
    )
    parser.add_argument("--capacity-warmup-epochs", type=int, default=0)
    parser.add_argument("--capacity-prior-tokenizer-init", action="store_true")
    parser.add_argument("--window-normalize-noncapacity", action="store_true")
    parser.add_argument("--initialize-observation-head-ridge", action="store_true")
    parser.add_argument("--minimum-checkpoint-epoch", type=int, default=0)
    parser.add_argument(
        "--validation-mode", choices=("shuffled", "cell"), default="shuffled"
    )
    parser.add_argument(
        "--capacity-reference", choices=("rated", "initial"), default="rated"
    )
    parser.add_argument(
        "--tju-capacity-input-scaling", choices=("c0", "minmax"), default="minmax"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Retrain requested folds even when matching completed results exist.",
    )
    parser.add_argument(
        "--tju-input-source", choices=("private", "aligned-reference"), default="private",
        help="Use the raw-reconstructed private input or the existing Wang-aligned 17-feature oracle.",
    )
    parser.add_argument(
        "--model-variant", choices=("full", "no-ct", "observation-aware"), default="full",
        help="Diagnostic architecture variant; no-ct matches the paper's CT ablation.",
    )
    args = parser.parse_args()
    write_json(args.output_root / "protocol.json", protocol_manifest())
    for dataset in args.datasets:
        # The paper reports only the first cell of each dataset.  The remaining
        # cells stay in the leave-one-cell-out training pool.
        requested = args.test_batteries or [PROTOCOLS[dataset].batteries[0]]
        batteries = [x for x in PROTOCOLS[dataset].batteries if x in requested]
        for battery in batteries:
            for seed in args.seeds:
                run_fold(
                    dataset, battery, seed, args.output_root,
                    torch.device(args.device), args.max_epochs, args.force,
                    args.tju_input_source,
                    args.model_variant,
                    args.data_version,
                    args.learning_rate,
                    args.patience,
                    args.batch_size,
                    args.validation_mode,
                    args.capacity_reference,
                    args.gradient_clip_norm,
                    args.lr_plateau_factor,
                    args.lr_plateau_patience,
                    args.initialize_head_to_target_median,
                    args.tju_capacity_input_scaling,
                    args.capacity_warmup_epochs,
                    args.capacity_prior_tokenizer_init,
                    args.window_normalize_noncapacity,
                    args.initialize_observation_head_ridge,
                    args.minimum_checkpoint_epoch,
                )


if __name__ == "__main__":
    main()
