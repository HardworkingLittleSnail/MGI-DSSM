"""Run PatchFormer and RUL-Mamba through their released forecasting interfaces.

Only the dataset/task contract is shared.  Target normalization, output
rescaling, loss, architecture and optimizer follow the released models.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparison_protocol import (  # noqa: E402
    DEFAULT_SEEDS,
    PROTOCOLS,
    evaluate_predictions,
    load_summary,
    seed_everything,
    write_csv,
    write_json,
)


@dataclass(frozen=True)
class NativeConfig:
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    gradient_clip: float
    build: dict[str, object]
    input_mode: str


def native_config(model_name: str, dataset: str, seq_len: int) -> NativeConfig:
    if model_name == "patchformer":
        return NativeConfig(
            1e-3, 16 if dataset == "nasa" else 128, 200, 10, 0.2,
            {
                "patch_len": 2, "seq_len": seq_len, "pred_len": 1,
                "enc_in": 1, "d_model": 16, "factor": 3,
                "dropout": 0.1, "output_attention": False,
                "n_heads": 8, "activation": "gelu", "e_layers": 2,
            },
            "univariate_capacity",
        )
    # RUL-Mamba was not evaluated on CALCE in the paper.  For the CALCE task
    # adaptation, retain the paper's closest univariate configuration (NASA):
    # Adam, SMAPE, lr=0.0022, batch=16, 200 epochs, patience=20 and clip=0.2.
    # Data splitting/normalization remain the shared leakage-safe contract.
    if model_name == "rul-mamba" and dataset == "calce":
        return NativeConfig(
            2.2e-3, 16, 200, 20, 0.2,
            {
                "seq_len": seq_len, "pred_len": 1, "enc_in": 1,
                "c_out": 1, "d_model": 48, "n_dec_layer": 1,
                "dropout": 0.0615, "expand": 2,
                "weight_decay": 0.0, "optimizer": "adam",
            },
            "univariate_capacity",
        )
    if dataset == "tju":
        return NativeConfig(
            1e-3, 128, 200, 20, 0.2,
            {
                "seq_len": seq_len, "pred_len": 1, "enc_in": 17,
                "c_out": 1, "d_model": 16, "n_dec_layer": 2,
                "dropout": 0.1, "expand": 2,
            },
            "paper_17_variable",
        )
    return NativeConfig(
        2.2e-3, 16, 200, 20, 0.2,
        {
            "seq_len": seq_len, "pred_len": 1, "enc_in": 1,
            "c_out": 1, "d_model": 48, "n_dec_layer": 1,
            "dropout": 0.0615, "expand": 2,
        },
        "univariate_capacity",
    )


def _split_index(length: int, seq_len: int) -> int:
    split = int(np.floor(length * 0.8))
    if split <= seq_len or length - split < 1:
        raise ValueError(f"series length {length} is too short for seq_len={seq_len}")
    return split


def _load_versioned_summary(dataset: str, data_version: str) -> dict[str, pd.DataFrame]:
    """Load an aligned versioned source without re-cleaning."""
    base = ROOT / "data" / ("version3" if data_version == "version3" else "processed-version2.0")
    protocol = PROTOCOLS[dataset]
    if dataset == "nasa":
        if data_version == "version3":
            payload = np.load(
                base / "NASA data" / "NASA_Data_minimal_interpolated.npy",
                allow_pickle=True,
            )[0]
            return {
                name: payload[name].copy().sort_values("Cycle").reset_index(drop=True)
                for name in protocol.batteries
            }
        result: dict[str, pd.DataFrame] = {}
        for name in protocol.batteries:
            structure = loadmat(
                base / "NASA data" / f"{name}.mat",
                squeeze_me=True,
                struct_as_record=False,
            )[name]
            capacity = np.asarray([
                float(record.data.Capacity)
                for record in np.atleast_1d(structure.cycle)
                if str(record.type).strip().lower() == "discharge"
            ], dtype=np.float64)
            result[name] = pd.DataFrame({
                "Cycle": np.arange(1, len(capacity) + 1, dtype=np.int64),
                "Capacity": capacity,
            })
        return result
    if dataset == "calce":
        payload = np.load(
            base / "CALCE data" / "CALCE_Data.npy", allow_pickle=True
        )[0]
        return {
            name: payload[name].copy().sort_values("Cycle").reset_index(drop=True)
            for name in protocol.batteries
        }
    if dataset == "tju":
        payload = np.load(
            base / "TJU data" / "Dataset_3_NCM_NCA_battery_1C.npy",
            allow_pickle=True,
        )[0]
        mapping = {"CY25-1": "CY25_1", "CY25-2": "CY25_2", "CY25-3": "CY25_3"}
        return {
            name: payload[source].copy().sort_values("Cycle").reset_index(drop=True)
            for name, source in mapping.items()
        }
    raise ValueError(f"{data_version} is not configured for {dataset}")


def _known_feature_payload(
    model_name: str, dataset: str, train_names: list[str], data_version: str,
):
    protocol = PROTOCOLS[dataset]
    frames = (
        _load_versioned_summary(dataset, data_version)
        if data_version in ("version2.0", "version3")
        else load_summary(protocol)
    )
    if model_name == "rul-mamba" and dataset == "tju":
        if data_version in ("version2.0", "version3"):
            indicators = [
                "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
                "CC Q", "CC charge time", "voltage slope", "voltage entropy",
                "current mean", "current std", "current kurtosis", "current skewness",
                "CV Q", "CV charge time", "current slope", "current entropy",
            ]
            payload = {
                name: {"features": np.column_stack((
                    frames[name][indicators].to_numpy(dtype=np.float64),
                    frames[name]["Capacity"].to_numpy(dtype=np.float64),
                ))}
                for name in protocol.batteries
            }
            version_root = "version3" if data_version == "version3" else "processed-version2.0"
            feature_source = f"data/{version_root}/TJU data/Dataset_3_NCM_NCA_battery_1C.npy"
        else:
            feature_path = ROOT / "data" / "processed" / "TJU data" / "batter_moe_17_features_aligned.npy"
            if not feature_path.exists():
                sys.path.insert(0, str(ROOT / "Compare-Models" / "MOE"))
                from prepare_tju_native_features import build
                build(ROOT)
            payload = np.load(feature_path, allow_pickle=True).item()
            feature_source = str(feature_path.relative_to(ROOT))
        names = [f"feature_{index:02d}" for index in range(17)]
        fit_rows = []
        for battery in train_names:
            values = np.asarray(payload[battery]["features"], dtype=np.float64)
            fit_rows.append(values[: _split_index(len(values), protocol.seq_len)])
        scaler = MinMaxScaler().fit(np.concatenate(fit_rows, axis=0))
        known = {battery: scaler.transform(np.asarray(payload[battery]["features"], dtype=np.float64))
                 for battery in protocol.batteries}
        return frames, names, known, {
            "type": "train-only feature-wise min-max",
            "minimum": scaler.data_min_.tolist(), "maximum": scaler.data_max_.tolist(),
            "source": feature_source,
        }

    fit_capacity = np.concatenate([
        frames[battery]["Capacity"].to_numpy(dtype=np.float64)[:
            _split_index(len(frames[battery]), protocol.seq_len)]
        for battery in train_names
    ]).reshape(-1, 1)
    scaler = MinMaxScaler().fit(fit_capacity)
    known = {
        battery: scaler.transform(frames[battery]["Capacity"].to_numpy(dtype=np.float64).reshape(-1, 1))
        for battery in protocol.batteries
    }
    return frames, ["Capacity"], known, {
        "type": "train-only min-max",
        "minimum_ah": float(scaler.data_min_[0]), "maximum_ah": float(scaler.data_max_[0]),
    }


def build_frames(model_name: str, dataset: str, test_battery: str, data_version: str):
    protocol = PROTOCOLS[dataset]
    train_names = [name for name in protocol.batteries if name != test_battery]
    frames, known_names, known_values, normalization = _known_feature_payload(
        model_name, dataset, train_names, data_version
    )

    def cell_frame(name: str, begin: int, end: int, reset_time: bool = False) -> pd.DataFrame:
        source = frames[name].iloc[begin:end]
        data = pd.DataFrame({
            "group_id": name,
            "time_idx": np.arange(len(source), dtype=np.int64) if reset_time else np.arange(begin, end, dtype=np.int64),
            "Cycle": source["Cycle"].to_numpy(dtype=np.int64),
            "capacity_ah": source["Capacity"].to_numpy(dtype=np.float64),
            "target": source["Capacity"].to_numpy(dtype=np.float64) / protocol.rated_capacity,
        })
        values = known_values[name][begin:end]
        for index, column in enumerate(known_names):
            data[column] = values[:, index]
        return data

    train_parts, validation_parts = [], []
    for name in train_names:
        length = len(frames[name])
        split = _split_index(length, protocol.seq_len)
        train_parts.append(cell_frame(name, 0, split))
        # The prefix supplies history only; the first validation target is the split point.
        validation_parts.append(cell_frame(name, split - protocol.seq_len, length, reset_time=True))
    train_frame = pd.concat(train_parts, ignore_index=True)
    validation_frame = pd.concat(validation_parts, ignore_index=True)
    test_frame = cell_frame(test_battery, 0, len(frames[test_battery]))
    return frames, train_names, known_names, train_frame, validation_frame, test_frame, normalization


def import_dependencies(model_name: str, dataset: str):
    import lightning.pytorch as pl
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import EncoderNormalizer, NaNLabelEncoder
    from pytorch_forecasting.metrics import SMAPE
    from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

    if model_name == "patchformer":
        sys.path.insert(0, str(ROOT / "Compare-Models" / "PatchFormer"))
        from ModelsModify.PatchFormer import PatchFormerNetModel as model_class
    else:
        sys.path.insert(0, str(ROOT / "Compare-Models" / "RUL-Mamba"))
        if dataset == "tju":
            from Models.RULMambaVAN import RULMambaVANNetModel as model_class
        else:
            from Models.RULMamba import RULMambaNetModel as model_class
    return pl, TimeSeriesDataSet, EncoderNormalizer, NaNLabelEncoder, SMAPE, Callback, EarlyStopping, ModelCheckpoint, model_class


def run_fold(model_name: str, dataset: str, test_battery: str, seed: int,
             output_root: Path, device: str | None, max_epochs: int | None,
             data_version: str, checkpoint_path: Path | None = None) -> None:
    protocol = PROTOCOLS[dataset]
    output = output_root / model_name / dataset / test_battery / f"seed_{seed}"
    result_path = output / "results.json"
    config = native_config(model_name, dataset, protocol.seq_len)
    epochs = int(max_epochs or config.max_epochs)
    patience = min(config.patience, epochs) if max_epochs else config.patience
    expected_native_config = {
        **asdict(config), "max_epochs": epochs, "patience": patience,
    }
    expected_processed_summary = (
        f"data/{'version3' if data_version == 'version3' else 'processed-version2.0'}"
        if data_version in ("version2.0", "version3")
        else str(protocol.summary_path.relative_to(ROOT))
    )
    if result_path.exists():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            recorded = existing.get("native_config", {})
            profile_matches = all(
                recorded.get(key) == value
                for key, value in expected_native_config.items()
            )
            source_matches = (
                existing.get("data_version") == data_version
                and existing.get("processed_summary") == expected_processed_summary
            )
            if existing.get("status") == "complete" and profile_matches and source_matches:
                print(f"skip complete: {output}", flush=True)
                return
            if existing.get("status") == "complete":
                print(f"rerun stale configuration: {output}", flush=True)
        except (OSError, json.JSONDecodeError):
            pass

    deps = import_dependencies(model_name, dataset)
    pl, TimeSeriesDataSet, EncoderNormalizer, NaNLabelEncoder, SMAPE, Callback, EarlyStopping, ModelCheckpoint, model_class = deps
    seed_everything(seed)
    pl.seed_everything(seed, workers=True)
    frames, train_names, known_names, train_frame, validation_frame, test_frame, normalization = build_frames(
        model_name, dataset, test_battery, data_version
    )

    dataset_kwargs = dict(
        time_idx="time_idx", target="target", group_ids=["group_id"],
        min_encoder_length=protocol.seq_len, max_encoder_length=protocol.seq_len,
        min_prediction_length=1, max_prediction_length=1,
        time_varying_known_reals=known_names, time_varying_unknown_reals=["target"],
        target_normalizer=EncoderNormalizer(), add_encoder_length=False,
        scalers={name: None for name in known_names},
        categorical_encoders={"group_id": NaNLabelEncoder(add_nan=True)},
    )
    training = TimeSeriesDataSet(train_frame, **dataset_kwargs)
    validation = TimeSeriesDataSet.from_dataset(training, validation_frame, stop_randomization=True)
    testing = TimeSeriesDataSet.from_dataset(training, test_frame, stop_randomization=True)
    train_loader = training.to_dataloader(train=True, batch_size=config.batch_size, shuffle=True, num_workers=0)
    validation_loader = validation.to_dataloader(train=False, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = testing.to_dataloader(train=False, batch_size=max(config.batch_size, 256), shuffle=False, num_workers=0)

    class History(Callback):
        def __init__(self):
            super().__init__()
            self.rows: list[dict[str, object]] = []

        def on_validation_epoch_end(self, trainer, _module):
            if trainer.sanity_checking:
                return
            metrics = trainer.callback_metrics
            row = {
                "epoch": int(trainer.current_epoch + 1),
                "train_loss": float(metrics["train_loss"].detach().cpu()) if "train_loss" in metrics else None,
                "validation_smape": float(metrics["val_loss"].detach().cpu()) if "val_loss" in metrics else None,
            }
            self.rows.append(row)
            train_text = "nan" if row["train_loss"] is None else f"{row['train_loss']:.8f}"
            val_text = "nan" if row["validation_smape"] is None else f"{row['validation_smape']:.8f}"
            print(
                f"[{model_name}/{dataset}/{test_battery}/seed={seed}] "
                f"epoch={row['epoch']}/{epochs} train={train_text} val={val_text}",
                flush=True,
            )

    output.mkdir(parents=True, exist_ok=True)
    history = History()
    checkpoint = ModelCheckpoint(
        dirpath=str(output / "checkpoints"), filename="best", monitor="val_loss", mode="min", save_top_k=1
    )
    early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=patience, mode="min")
    accelerator = "gpu" if (device is None and torch.cuda.is_available()) or (device and device.startswith("cuda")) else "cpu"
    trainer = pl.Trainer(
        max_epochs=epochs, accelerator=accelerator, devices=1,
        gradient_clip_val=config.gradient_clip, callbacks=[history, checkpoint, early_stop],
        logger=False, default_root_dir=str(output), deterministic=True,
        enable_model_summary=False, enable_progress_bar=False,
    )
    model = model_class.from_dataset(
        training, **config.build, learning_rate=config.learning_rate, loss=SMAPE()
    )
    started = time.time()
    if checkpoint_path is None:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
        selected_checkpoint = Path(checkpoint.best_model_path)
    else:
        selected_checkpoint = checkpoint_path.resolve()
        if not selected_checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {selected_checkpoint}")
    best_model = model_class.load_from_checkpoint(str(selected_checkpoint), map_location="cpu")
    predictions = best_model.predict(test_loader, batch_size=max(config.batch_size, 256))
    y_pred = predictions.detach().cpu().numpy().reshape(-1) * protocol.rated_capacity
    y_true = test_frame["capacity_ah"].to_numpy(dtype=np.float64)[protocol.seq_len:]
    cycles = test_frame["Cycle"].to_numpy(dtype=np.int64)[protocol.seq_len:]
    if len(y_pred) != len(y_true):
        raise RuntimeError(f"prediction alignment failed: pred={len(y_pred)} true={len(y_true)}")
    metrics, prediction_rows = evaluate_predictions(cycles, y_true, y_pred, protocol)
    for row in prediction_rows:
        row.update({"model": model_name, "dataset": dataset, "battery": test_battery, "seed": seed})
    best_val = min((row["validation_smape"] for row in history.rows if row["validation_smape"] is not None), default=None)
    best_epoch = next((row["epoch"] for row in history.rows if row["validation_smape"] == best_val), None)
    result = {
        "status": "complete", "model": model_name, "dataset": dataset,
        "test_battery": test_battery, "train_batteries": train_names, "seed": seed,
        "task": f"{protocol.seq_len}-cycle rolling one-step capacity prediction",
        "start_points": list(protocol.start_points), "metrics": metrics,
        "best_epoch": best_epoch, "best_validation_smape": best_val,
        "checkpoint_source": str(selected_checkpoint),
        "elapsed_seconds": time.time() - started,
        "normalization": normalization,
        "target_contract": "C/C0 + EncoderNormalizer + transform_output; metrics in Ah",
        "native_config": {**asdict(config), "max_epochs": epochs, "patience": patience},
        "data_version": data_version,
        "processed_summary": expected_processed_summary,
    }
    write_csv(output / "training_history.csv", history.rows)
    write_csv(output / "predictions.csv", prediction_rows)
    write_json(result_path, result)
    print(f"complete: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("patchformer", "rul-mamba"), required=True)
    parser.add_argument("--datasets", nargs="+", choices=tuple(PROTOCOLS), default=list(PROTOCOLS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "comparison_models_native_10seeds")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--data-version", choices=("processed", "version2.0", "version3"),
        default="version3",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None,
                        help="Evaluate an existing compatible checkpoint without retraining.")
    args = parser.parse_args()
    for dataset in args.datasets:
        test_battery = PROTOCOLS[dataset].batteries[0]
        for seed in args.seeds:
            run_fold(
                args.model, dataset, test_battery, seed, args.output_root.resolve(),
                args.device, args.max_epochs, args.data_version, args.checkpoint_path,
            )


if __name__ == "__main__":
    main()
