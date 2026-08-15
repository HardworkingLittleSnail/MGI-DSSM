"""Run RUL-Mamba on CALCE with a selectable experiment protocol.

The RUL-Mamba architecture, univariate capacity input, direct 64-to-1
forecasting flow, EncoderNormalizer and SMAPE objective are intentionally kept
unchanged. Dataset preprocessing, input Min-Max fitting scope and training
settings are controlled by YAML so comparison and native RUL-Mamba protocols
can coexist without overwriting one another.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

# pytorch-forecasting 0.10.3 uses aliases removed in recent NumPy releases.
for _alias, _type in {"float": float, "int": int, "bool": bool}.items():
    if _alias not in np.__dict__:
        setattr(np, _alias, _type)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "Configs/CALCE/Univariable/Base_MGI_Aligned.yaml"),
    )
    parser.add_argument(
        "--model-config",
        default=str(PROJECT_ROOT / "Configs/CALCE/Univariable/RULMamba_MGI_Aligned.yaml"),
    )
    parser.add_argument("--batteries", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--defer-summary",
        action="store_true",
        help="Save per-run artifacts only; build aggregate files in a later resume pass.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def save_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)


def save_yaml(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, allow_unicode=True, sort_keys=False)


def set_seed(seed: int, torch_module) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed(seed)
        torch_module.cuda.manual_seed_all(seed)
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True


def configure_environment(config: dict, args: argparse.Namespace) -> None:
    cache_root = PROJECT_ROOT / ".cache"
    matplotlib_cache = cache_root / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    gpu_id = args.gpu_id if args.gpu_id is not None else config["runtime"].get("gpu_id")
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def load_dependencies() -> SimpleNamespace:
    import lightning.pytorch as pl
    import torch
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import EncoderNormalizer
    from pytorch_forecasting.metrics import MAE, SMAPE
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.metrics import r2_score

    return SimpleNamespace(
        pl=pl,
        torch=torch,
        TimeSeriesDataSet=TimeSeriesDataSet,
        EncoderNormalizer=EncoderNormalizer,
        MAE=MAE,
        SMAPE=SMAPE,
        EarlyStopping=EarlyStopping,
        ModelCheckpoint=ModelCheckpoint,
        r2_score=r2_score,
    )


def setup_legacy_module_aliases() -> None:
    models_pkg = importlib.import_module("Models")
    layers_pkg = importlib.import_module("Models.Layers")
    sys.modules.setdefault("models", models_pkg)
    sys.modules.setdefault("models.layers", layers_pkg)
    sys.modules.setdefault("ModelsModify", models_pkg)
    sys.modules.setdefault("ModelsModify.layers", layers_pkg)
    sys.modules.setdefault("layers", layers_pkg)


def import_object(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def load_calce_cache(path: Path, battery_names: list[str]) -> dict[str, pd.DataFrame]:
    raw = np.load(path, allow_pickle=True)
    if not isinstance(raw, np.ndarray) or raw.size != 1:
        raise ValueError(f"Unexpected CALCE cache container: shape={getattr(raw, 'shape', None)}")
    payload = raw.reshape(-1)[0]
    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()
    if not isinstance(payload, dict):
        raise TypeError(f"Expected battery dictionary, got {type(payload)!r}")

    output: dict[str, pd.DataFrame] = {}
    for name in battery_names:
        if name not in payload:
            raise KeyError(f"Battery {name} is missing from {path}")
        frame = payload[name][["BatteryName", "Cycle", "Capacity"]].copy()
        frame["BatteryName"] = name
        frame["Cycle"] = pd.to_numeric(frame["Cycle"], errors="raise").astype(int)
        frame["Capacity"] = pd.to_numeric(frame["Capacity"], errors="coerce").astype(float)
        frame.sort_values("Cycle", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        expected_cycles = np.arange(1, len(frame) + 1)
        if not np.array_equal(frame["Cycle"].to_numpy(), expected_cycles):
            raise ValueError(f"{name} cycles are not exactly 1..{len(frame)}")
        output[name] = frame
    return output


def preprocess_calce_batteries(
    batteries: dict[str, pd.DataFrame], method: str
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Apply the selected preprocessing independently to each battery.

    ``rul_mamba_paper`` mirrors the original RUL-Mamba data pipeline: linear
    interpolation, global two-standard-deviation outlier removal, followed by
    another interpolation. No held-out battery statistics are shared here.
    """
    processed: dict[str, pd.DataFrame] = {}
    stats: dict[str, dict] = {}
    for name, source in batteries.items():
        frame = source.copy()
        capacity = pd.to_numeric(frame["Capacity"], errors="coerce").astype(float)
        missing_before = int(capacity.isna().sum())

        if method == "raw":
            cleaned = capacity
            outlier_count = 0
        elif method == "rul_mamba_paper":
            interpolated = capacity.interpolate(
                method="linear", limit_direction="both"
            )
            mean = float(interpolated.mean())
            std = float(interpolated.std(ddof=0))
            outlier_mask = (
                (interpolated - mean).abs() > 2.0 * std
                if np.isfinite(std) and std > 0.0
                else pd.Series(False, index=interpolated.index)
            )
            outlier_count = int(outlier_mask.sum())
            cleaned = interpolated.mask(outlier_mask).interpolate(
                method="linear", limit_direction="both"
            )
        else:
            raise ValueError(f"Unsupported CALCE preprocessing method: {method!r}")

        if not np.isfinite(cleaned.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} has non-finite capacities after {method}")
        frame["Capacity"] = cleaned.to_numpy(dtype=float)
        processed[name] = frame
        stats[name] = {
            "method": method,
            "missing_before": missing_before,
            "two_sigma_outliers_replaced": outlier_count,
        }
    return processed, stats


def _format_frame(
    frame: pd.DataFrame,
    begin: int,
    end: int,
    rated_capacity: float,
    capacity_min: float,
    capacity_max: float,
) -> pd.DataFrame:
    output = frame.iloc[begin:end].copy()
    output["target"] = output["Capacity"] / rated_capacity
    output["CapacityAh"] = output["Capacity"]
    output["Capacity"] = (output["target"] - capacity_min) / (capacity_max - capacity_min)
    output["time_idx"] = np.arange(len(output), dtype=np.int64)
    return output[["BatteryName", "time_idx", "Cycle", "Capacity", "target", "CapacityAh"]]


def prepare_fold(
    batteries: dict[str, pd.DataFrame],
    test_name: str,
    start_point: int,
    seq_len: int,
    validation_fraction: float,
    rated_capacity: float,
    normalization_scope: str = "train_prefix",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train_names = [name for name in batteries if name != test_name]
    splits = {
        name: int(len(batteries[name]) * (1.0 - validation_fraction))
        for name in train_names
    }
    if normalization_scope == "train_prefix":
        normalization_frames = [
            batteries[name].iloc[: splits[name]] for name in train_names
        ]
    elif normalization_scope == "full_held_in_batteries":
        # This matches RUL-Mamba's original ordering: fit input Min-Max bounds
        # on the selected training batteries, then make the 80/20 split. The
        # held-out test battery is never used to fit these bounds.
        normalization_frames = [batteries[name] for name in train_names]
    else:
        raise ValueError(f"Unsupported normalization scope: {normalization_scope!r}")
    normalization_values = np.concatenate([
        frame["Capacity"].to_numpy(dtype=float) / rated_capacity
        for frame in normalization_frames
    ])
    capacity_min = float(normalization_values.min())
    capacity_max = float(normalization_values.max())
    if capacity_max <= capacity_min:
        raise ValueError("Training-only input Min-Max bounds are invalid")

    train_frames = []
    val_frames = []
    for name in train_names:
        split = splits[name]
        train_frames.append(
            _format_frame(
                batteries[name], 0, split, rated_capacity, capacity_min, capacity_max
            )
        )
        val_frames.append(
            _format_frame(
                batteries[name], split - seq_len, len(batteries[name]),
                rated_capacity, capacity_min, capacity_max,
            )
        )

    test_begin = start_point - 1 - seq_len
    if test_begin < 0:
        raise ValueError(f"SP {start_point} has fewer than {seq_len} history cycles")
    test_frame = _format_frame(
        batteries[test_name], test_begin, len(batteries[test_name]),
        rated_capacity, capacity_min, capacity_max,
    )
    train_frame = pd.concat(train_frames, ignore_index=True)
    val_frame = pd.concat(val_frames, ignore_index=True)
    expected = {
        "train_names": train_names,
        "split_index_by_battery": splits,
        "input_capacity_min_soh": capacity_min,
        "input_capacity_max_soh": capacity_max,
        "normalization_scope": normalization_scope,
        "expected_train_windows": int(sum(splits[name] - seq_len for name in train_names)),
        "expected_validation_windows": int(
            sum(len(batteries[name]) - splits[name] for name in train_names)
        ),
        "expected_test_windows": int(len(batteries[test_name]) - start_point + 1),
        "first_test_target_cycle": int(start_point),
        "last_test_target_cycle": int(len(batteries[test_name])),
    }
    return train_frame, val_frame, test_frame, expected


def build_dataset(frame: pd.DataFrame, config: dict, deps: SimpleNamespace):
    return deps.TimeSeriesDataSet(
        frame,
        time_idx="time_idx",
        target="target",
        group_ids=["BatteryName"],
        min_encoder_length=config["window"]["seq_len"],
        max_encoder_length=config["window"]["seq_len"],
        min_prediction_length=config["window"]["pred_len"],
        max_prediction_length=config["window"]["pred_len"],
        time_varying_known_reals=["Capacity"],
        time_varying_unknown_reals=["target"],
        target_normalizer=deps.EncoderNormalizer(),
        scalers={"Capacity": None},
        add_encoder_length=False,
    )


def make_loaders(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    config: dict,
    deps: SimpleNamespace,
):
    batch_size = config["train"]["batch_size"]
    workers = config["train"]["num_workers"]
    train_ds = build_dataset(train_frame, config, deps)
    val_ds = build_dataset(val_frame, config, deps)
    test_ds = build_dataset(test_frame, config, deps)
    train_loader = train_ds.to_dataloader(
        train=True, batch_size=batch_size,
        shuffle=config["train"].get("shuffle_train", True),
        num_workers=workers,
        drop_last=config["train"].get("drop_last_train", False),
    )
    val_loader = val_ds.to_dataloader(
        train=False, batch_size=batch_size, shuffle=False,
        num_workers=workers, drop_last=False,
    )
    test_loader = test_ds.to_dataloader(
        train=False, batch_size=batch_size, shuffle=False,
        num_workers=workers, drop_last=False,
    )
    return train_ds, train_loader, val_loader, test_loader, (len(train_ds), len(val_ds), len(test_ds))


def load_model_checkpoint(model_class, checkpoint_path: str, torch_module, device):
    original_load = torch_module.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        try:
            return original_load(*args, **kwargs)
        except TypeError:
            kwargs.pop("weights_only", None)
            return original_load(*args, **kwargs)

    torch_module.load = trusted_load
    try:
        return model_class.load_from_checkpoint(checkpoint_path).to(device=device)
    finally:
        torch_module.load = original_load


def patchformer_rul_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict:
    # The final MGI-DSSM 40-run artifacts and summary use the symmetric
    # consecutive-threshold rule for both curves. The repository's current
    # metrics.py later diverged to an asymmetric predicted crossing; using that
    # newer function does not reproduce the saved final benchmark (notably
    # CS2_36 and CS2_38). This implements the protocol that generated it.
    def consecutive_crossing(values: np.ndarray) -> int:
        for index in range(len(values) - 1):
            if values[index] <= threshold and values[index + 1] <= threshold:
                return index - 1
        return len(values)

    true_re = consecutive_crossing(y_true)
    pred_re = consecutive_crossing(y_pred)
    rul_real = true_re + 1
    rul_pred = pred_re + 1
    ae = abs(true_re - pred_re)
    # The final benchmark divides by the zero-based true crossing index, not
    # by the reported one-based RUL. This is visible in the saved CS2_36/seed 7
    # value: 8 / 445 = 0.017977528089887642.
    re_score = min(ae / max(abs(true_re), 1), 1.0)
    return {
        "rul_real": int(rul_real),
        "rul_pred": int(rul_pred),
        "ae": int(ae),
        "re": float(re_score),
    }


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    persistence: np.ndarray,
    rated_capacity: float,
    deps: SimpleNamespace,
) -> dict:
    if not (len(y_true) == len(y_pred) == len(persistence)):
        raise ValueError(
            f"Metric lengths differ: true={len(y_true)}, pred={len(y_pred)}, "
            f"persistence={len(persistence)}"
        )
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("Predictions or labels contain non-finite values")
    metrics = {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean(np.square(y_true - y_pred)))),
        "r2": float(deps.r2_score(y_true, y_pred)),
        "persistence_mae": float(np.mean(np.abs(y_true - persistence))),
    }
    metrics.update(patchformer_rul_metrics(y_true, y_pred, rated_capacity * 0.7))
    return metrics


def result_complete(run_dir: Path) -> bool:
    required = ["Metrics.json", "Prediction.npy", "Actual.npy", "predictions.csv"]
    if not all((run_dir / name).exists() for name in required):
        return False
    try:
        with open(run_dir / "Metrics.json", "r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload.get("eol_protocol") == "symmetric_consecutive_below_70_percent"
    except (OSError, json.JSONDecodeError):
        return False


def run_one(
    battery: str,
    seed: int,
    batteries: dict[str, pd.DataFrame],
    config: dict,
    output_root: Path,
    log_root: Path,
    model_class,
    deps: SimpleNamespace,
    force: bool,
) -> dict:
    run_dir = output_root / battery / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if result_complete(run_dir) and not force:
        with open(run_dir / "Metrics.json", "r", encoding="utf-8") as file:
            return json.load(file)

    set_seed(seed, deps.torch)
    start_point = int(config["dataset"]["start_points"][battery])
    train_frame, val_frame, test_frame, protocol_info = prepare_fold(
        batteries=batteries,
        test_name=battery,
        start_point=start_point,
        seq_len=config["window"]["seq_len"],
        validation_fraction=config["train"]["validation_fraction"],
        rated_capacity=config["dataset"]["rated_capacity"],
        normalization_scope=config.get("protocol", {}).get(
            "normalization_scope", "train_prefix"
        ),
    )
    training, train_loader, val_loader, test_loader, dataset_counts = make_loaders(
        train_frame, val_frame, test_frame, config, deps
    )
    expected_counts = (
        protocol_info["expected_train_windows"],
        protocol_info["expected_validation_windows"],
        protocol_info["expected_test_windows"],
    )
    if dataset_counts != expected_counts:
        raise AssertionError(f"Window count mismatch: actual={dataset_counts}, expected={expected_counts}")

    build_args = config["model"]["build_args"]
    model = model_class.from_dataset(
        training,
        seq_len=build_args["seq_len"],
        pred_len=build_args["pred_len"],
        enc_in=build_args["enc_in"],
        c_out=build_args["c_out"],
        d_model=build_args["d_model"],
        n_dec_layer=build_args["n_dec_layer"],
        dropout=build_args["dropout"],
        expand=build_args["expand"],
        learning_rate=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
        optimizer=config["train"]["optimizer"],
        loss=deps.StableSMAPE(),
        logging_metrics=deps.torch.nn.ModuleList([deps.MAE()]),
        # PF 0.10.3 returns an invalid empty scheduler dictionary when this is
        # None under Lightning 1.9. A value above max_epochs preserves the
        # final MGI-DSSM protocol's fixed learning rate without ever stepping.
        reduce_on_plateau_patience=1000,
    )

    checkpoint_monitor = config["train"].get("checkpoint_monitor", "val_MAE")
    checkpoint_filename = (
        "{epoch:02d}-{val_loss:.8f}"
        if checkpoint_monitor == "val_loss"
        else "{epoch:02d}-{val_MAE:.8f}"
    )
    checkpoint = deps.ModelCheckpoint(
        dirpath=str(run_dir / "Checkpoints"),
        filename=checkpoint_filename,
        monitor=checkpoint_monitor,
        mode="min",
        save_top_k=1,
    )
    early_stop = deps.EarlyStopping(
        monitor=checkpoint_monitor,
        min_delta=config["train"].get(
            "early_stopping_min_delta",
            1e-6 / config["dataset"]["rated_capacity"],
        ),
        patience=config["train"]["patience"],
        mode="min",
        verbose=False,
    )

    class NumericalGradientGuard(deps.pl.Callback):
        """Prevent a divergent batch from corrupting the saved finite model."""

        def on_after_backward(self, trainer, pl_module) -> None:
            non_finite = 0
            for parameter in pl_module.parameters():
                gradient = parameter.grad
                if gradient is not None and not deps.torch.isfinite(gradient).all():
                    non_finite += int((~deps.torch.isfinite(gradient)).sum().item())
                    deps.torch.nan_to_num_(
                        gradient, nan=0.0, posinf=0.0, neginf=0.0
                    )
            if non_finite:
                print(
                    f"Numerical guard stopped training at epoch={trainer.current_epoch} "
                    f"global_step={trainer.global_step}; "
                    f"sanitized_gradient_values={non_finite}",
                    flush=True,
                )
                trainer.should_stop = True

    trainer = deps.pl.Trainer(
        max_epochs=config["train"]["max_epochs"],
        gradient_clip_val=config["train"]["gradient_clip_val"],
        callbacks=[early_stop, checkpoint, NumericalGradientGuard()],
        logger=False,
        default_root_dir=str(run_dir),
        accelerator="gpu" if deps.torch.cuda.is_available() else "cpu",
        devices=1,
        deterministic=True,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )

    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{battery}_seed_{seed}.log"
    with open(log_path, "w", encoding="utf-8") as log:
        def write(message: str) -> None:
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        write(f"battery={battery} seed={seed} SP={start_point}")
        write(f"train_batteries={protocol_info['train_names']}")
        write(f"windows train/val/test={dataset_counts}")
        write(f"params={model.size()} architecture={build_args}")
        train_start = time.time()
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("RUL-Mamba has no trainable parameters before Trainer.fit")
        try:
            # Keep the training boundary explicit. This is important when this
            # entry point is called from a long-lived process with legacy
            # Lightning/PyTorch-Forecasting versions that also run inference
            # under torch.no_grad().
            with deps.torch.enable_grad():
                trainer.fit(
                    model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                )
        except Exception:
            write("TRAINING_ERROR\n" + traceback.format_exc())
            raise
        train_seconds = time.time() - train_start
        if not checkpoint.best_model_path:
            raise RuntimeError("No best checkpoint was produced")

        device = deps.torch.device("cuda" if deps.torch.cuda.is_available() else "cpu")
        best_model = load_model_checkpoint(
            model_class, checkpoint.best_model_path, deps.torch, device
        )
        infer_start = time.time()
        predictions = best_model.predict(test_loader, batch_size=config["train"]["batch_size"])
        infer_seconds = time.time() - infer_start
        y_pred = predictions.detach().cpu().numpy().reshape(-1) * config["dataset"]["rated_capacity"]
        y_true = test_frame.iloc[config["window"]["seq_len"]:]["CapacityAh"].to_numpy(dtype=float)
        persistence = test_frame.iloc[
            config["window"]["seq_len"] - 1 : -1
        ]["CapacityAh"].to_numpy(dtype=float)
        cycles = test_frame.iloc[config["window"]["seq_len"]:]["Cycle"].to_numpy(dtype=int)
        metrics = evaluate(
            y_true, y_pred, persistence, config["dataset"]["rated_capacity"], deps
        )
        best_epoch = int(deps.torch.load(
            checkpoint.best_model_path, map_location="cpu", weights_only=False
        )["epoch"])
        metrics.update(
            {
                "battery": battery,
                "seed": int(seed),
                "start_point": start_point,
                "num_windows": int(len(y_true)),
                "train_windows": int(dataset_counts[0]),
                "validation_windows": int(dataset_counts[1]),
                "best_epoch": best_epoch,
                "stopped_epoch": int(trainer.current_epoch),
                "train_seconds": float(train_seconds),
                "infer_seconds": float(infer_seconds),
                "best_model_path": str(Path(checkpoint.best_model_path).resolve()),
                "eol_protocol": "symmetric_consecutive_below_70_percent",
                "protocol": protocol_info,
            }
        )
        write(
            f"MAE={metrics['mae']:.8f} RMSE={metrics['rmse']:.8f} "
            f"R2={metrics['r2']:.8f} RUL={metrics['rul_real']}/{metrics['rul_pred']} "
            f"AE={metrics['ae']} RE={metrics['re']:.8f} epoch={best_epoch}"
        )

    np.save(run_dir / "Prediction.npy", y_pred)
    np.save(run_dir / "Actual.npy", y_true)
    np.save(run_dir / "Persistence.npy", persistence)
    with open(run_dir / "predictions.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["battery", "cycle", "actual_ah", "prediction_ah", "persistence_ah"])
        writer.writerows(zip([battery] * len(cycles), cycles, y_true, y_pred, persistence))
    save_json(metrics, run_dir / "Metrics.json")
    return metrics


def summarize(rows: list[dict], config: dict) -> dict:
    metric_keys = ["mae", "rmse", "r2", "ae", "re", "persistence_mae"]
    by_battery = {}
    for battery in config["dataset"]["battery_list"]:
        selected = [row for row in rows if row["battery"] == battery]
        if not selected:
            continue
        entry = {"runs": len(selected)}
        for key in metric_keys:
            values = np.asarray([row[key] for row in selected], dtype=float)
            entry[key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "best": float(values.max() if key == "r2" else values.min()),
                "best_seed": int(
                    selected[int(np.argmax(values) if key == "r2" else np.argmin(values))]["seed"]
                ),
            }
        best_complete = min(selected, key=lambda row: row["rmse"])
        entry["best_complete_by_rmse"] = best_complete
        by_battery[battery] = entry

    macro_runs = []
    for seed in config["train"]["seeds"]:
        selected = [row for row in rows if row["seed"] == seed]
        if len(selected) != len(config["dataset"]["battery_list"]):
            continue
        macro_runs.append(
            {
                "seed": int(seed),
                **{
                    key: float(np.mean([row[key] for row in selected]))
                    for key in metric_keys
                },
            }
        )
    macro = {"runs": macro_runs}
    for key in metric_keys:
        values = np.asarray([row[key] for row in macro_runs], dtype=float)
        if values.size:
            best_index = int(np.argmax(values) if key == "r2" else np.argmin(values))
            macro[key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "best": float(values[best_index]),
                "best_seed": int(macro_runs[best_index]["seed"]),
            }
    if macro_runs:
        macro["best_complete_by_rmse"] = min(macro_runs, key=lambda row: row["rmse"])
    return {"by_battery": by_battery, "macro_across_batteries_by_seed": macro}


def save_all_results(rows: list[dict], config: dict, output_root: Path) -> None:
    sorted_rows = sorted(rows, key=lambda row: (row["battery"], row["seed"]))
    save_json(sorted_rows, output_root / "all_results.json")
    columns = [
        "battery", "seed", "start_point", "num_windows", "mae", "rmse", "r2",
        "rul_real", "rul_pred", "ae", "re", "persistence_mae", "best_epoch",
        "stopped_epoch", "train_seconds", "infer_seconds", "best_model_path",
    ]
    with open(output_root / "all_results.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted_rows)
    save_json(summarize(sorted_rows, config), output_root / "Summary.json")


def verify_protocol(batteries: dict[str, pd.DataFrame], config: dict) -> list[dict]:
    rows = []
    for battery in config["dataset"]["battery_list"]:
        sp = int(config["dataset"]["start_points"][battery])
        train, val, test, info = prepare_fold(
            batteries, battery, sp, config["window"]["seq_len"],
            config["train"]["validation_fraction"], config["dataset"]["rated_capacity"],
            config.get("protocol", {}).get("normalization_scope", "train_prefix"),
        )
        true_values = test.iloc[config["window"]["seq_len"]:]["CapacityAh"].to_numpy()
        true_rul = patchformer_rul_metrics(
            true_values, np.full_like(true_values, 999.0),
            config["dataset"]["rated_capacity"] * 0.7,
        )["rul_real"]
        row = {
            "battery": battery,
            "length": len(batteries[battery]),
            "SP": sp,
            "train_windows": info["expected_train_windows"],
            "validation_windows": info["expected_validation_windows"],
            "test_windows": info["expected_test_windows"],
            "true_rul": true_rul,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    return rows


def main() -> None:
    args = parse_args()
    config = deep_merge(load_yaml(args.config), load_yaml(args.model_config))
    run_batteries = (
        list(args.batteries)
        if args.batteries is not None
        else list(config["dataset"]["battery_list"])
    )
    unknown_batteries = set(run_batteries) - set(config["dataset"]["battery_list"])
    if unknown_batteries:
        raise ValueError(f"Unknown requested batteries: {sorted(unknown_batteries)}")
    if args.seeds is not None:
        config["train"]["seeds"] = args.seeds
    if args.max_epochs is not None:
        config["train"]["max_epochs"] = args.max_epochs
    configure_environment(config, args)

    batteries = load_calce_cache(
        PROJECT_ROOT / config["dataset"]["battery_cache_path"],
        config["dataset"]["battery_list"],
    )
    preprocessing_method = config.get("protocol", {}).get("preprocessing", "raw")
    batteries, preprocessing_stats = preprocess_calce_batteries(
        batteries, preprocessing_method
    )
    print("preprocessing=" + json.dumps(preprocessing_stats, ensure_ascii=False))
    verify_protocol(batteries, config)
    if args.verify_only:
        return

    deps = load_dependencies()
    setup_legacy_module_aliases()
    model_class = import_object(config["model"]["class_path"])
    stable_smape_class = import_object("Models.Metrics.StableSMAPE")
    deps.StableSMAPE = stable_smape_class
    output_root = PROJECT_ROOT / config["output"]["outputs_dir"]
    log_root = PROJECT_ROOT / config["output"]["logs_dir"]
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.defer_summary:
        save_yaml(config, output_root / "Merged_Config.yaml")

    rows = []
    total = len(run_batteries) * len(config["train"]["seeds"])
    completed = 0
    for battery in run_batteries:
        for seed in config["train"]["seeds"]:
            completed += 1
            print(f"\n[{completed}/{total}] battery={battery} seed={seed}", flush=True)
            row = run_one(
                battery, int(seed), batteries, config, output_root, log_root,
                model_class, deps, args.force,
            )
            rows.append(row)
            if not args.defer_summary:
                save_all_results(rows, config, output_root)

    if args.defer_summary:
        print(f"Completed {len(rows)} runs. Aggregate summary deferred.")
    else:
        print(f"Completed {len(rows)} runs. Summary: {output_root / 'Summary.json'}")


if __name__ == "__main__":
    main()
