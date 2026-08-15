"""Train Autoformer and iTransformer under the aligned battery benchmark.

The dataset/task contract is shared with the other comparison models.  The
architectural ideas remain model-native: Autoformer uses progressive series
decomposition and FFT Auto-Correlation; iTransformer embeds each degradation
series as a variate token and attends across those tokens.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparison_protocol import (  # noqa: E402
    DEFAULT_SEEDS,
    PROTOCOLS,
    chronological_samples,
    evaluate_predictions,
    load_summary,
    protocol_manifest,
    seed_everything,
    write_csv,
    write_json,
)


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    gradient_clip: float | None
    build: dict[str, object]
    input_features: tuple[str, ...]


def model_config(model_name: str, dataset: str, seq_len: int) -> TrainConfig:
    """Model-native settings adapted to the current battery protocol."""
    batch_size = 63 if dataset == "nasa" else 128
    max_epochs = 200
    patience = 20
    if model_name == "autoformer":
        # An odd kernel is required by the original centered moving average.
        moving_avg = 5 if seq_len == 16 else 25
        return TrainConfig(5e-4, batch_size, max_epochs, patience, 0.2, {
            "seq_len": seq_len,
            "label_len": max(1, seq_len // 2),
            "pred_len": 1,
            "enc_in": 1,
            "dec_in": 1,
            "c_out": 1,
            "d_model": 32,
            "n_heads": 4,
            "e_layers": 2,
            "d_layers": 1,
            "d_ff": 128,
            "moving_avg": moving_avg,
            "factor": 1,
            "dropout": 0.1,
            "embed": "timeF",
            "freq": "m",
            "activation": "gelu",
            "output_attention": False,
        }, ("capacity",))
    # The official iTransformer runner defaults to patience=3.  Keeping the
    # battery-specific patience=20 allowed the smooth TJU series to train for
    # well over 100 epochs and materially departed from the author protocol.
    return TrainConfig(1e-4, 32, max_epochs, 3, None, {
        "seq_len": seq_len,
        "pred_len": 1,
        "d_model": 64,
        "n_heads": 4,
        "e_layers": 2,
        "d_ff": 128,
        "factor": 1,
        "dropout": 0.1,
        "embed": "timeF",
        "freq": "m",
        "activation": "gelu",
        "output_attention": False,
        "use_norm": True,
        "class_strategy": "projection",
        # Audit marker only; the author trainer optimizes one unweighted MSE
        # over every forecast variate (Exp_Long_Term_Forecast._select_criterion).
        "loss_contract": "author_unweighted_joint_mse",
    }, ("capacity", "delta"))


def load_versioned_summary(dataset: str, data_version: str) -> dict[str, pd.DataFrame]:
    """Load an aligned versioned source without applying any new cleaning."""
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
                "Cycle": np.arange(1, len(capacity) + 1), "Capacity": capacity,
            })
        return result
    if dataset == "calce":
        payload = np.load(base / "CALCE data" / "CALCE_Data.npy", allow_pickle=True)[0]
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


def load_frames(dataset: str, data_version: str) -> dict[str, pd.DataFrame]:
    frames = (
        load_versioned_summary(dataset, data_version)
        if data_version in ("version2.0", "version3")
        else load_summary(PROTOCOLS[dataset])
    )
    for name, frame in frames.items():
        if not {"Cycle", "Capacity"}.issubset(frame.columns):
            raise ValueError(f"{dataset}/{name} lacks Cycle or Capacity")
        if not np.isfinite(frame["Capacity"].to_numpy(dtype=float)).all():
            raise ValueError(f"{dataset}/{name} contains non-finite capacity")
    return frames


def degradation_features(capacity: np.ndarray) -> np.ndarray:
    """Causal variates: level, first difference, and two trailing trends."""
    capacity = np.asarray(capacity, dtype=np.float32)
    delta = np.diff(capacity, prepend=capacity[0])

    def trailing_mean(window: int) -> np.ndarray:
        return pd.Series(capacity).rolling(window, min_periods=1).mean().to_numpy(dtype=np.float32)

    return np.column_stack((capacity, delta, trailing_mean(3), trailing_mean(7))).astype(np.float32)


class BatteryWindows(Dataset):
    def __init__(
        self,
        model_name: str,
        values: dict[str, np.ndarray],
        cycles: dict[str, np.ndarray],
        samples: list[tuple[str, int]],
        seq_len: int,
    ) -> None:
        self.model_name = model_name
        self.values = values
        self.cycles = cycles
        self.samples = samples
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        name, target = self.samples[index]
        x = self.values[name][target - self.seq_len:target]
        y = self.values[name][target]
        if self.model_name == "autoformer":
            x = x[:, :1]
            y = y[:1]
        # A single monotone cycle coordinate is a valid timeF marker (freq=m).
        cycle = self.cycles[name].astype(np.float32)
        # Fixed unit conversion avoids revealing a held-out cell's final cycle.
        x_mark = (cycle[target - self.seq_len:target] / 1000.0)[:, None]
        y_mark = np.asarray([[cycle[target] / 1000.0]], dtype=np.float32)
        return tuple(torch.as_tensor(v, dtype=torch.float32) for v in (x, y, x_mark, y_mark))


def import_model(model_name: str):
    source = HERE / ("Autoformer" if model_name == "autoformer" else "iTransformer")
    sys.path.insert(0, str(source))
    module_name = "models.Autoformer" if model_name == "autoformer" else "model.iTransformer"
    try:
        return importlib.import_module(module_name).Model
    finally:
        sys.path.remove(str(source))


def build_model(model_name: str, config: TrainConfig) -> torch.nn.Module:
    return import_model(model_name)(SimpleNamespace(**config.build))


def forward_model(model_name: str, model: torch.nn.Module, batch, config: TrainConfig):
    x, y, x_mark, y_mark = batch
    if model_name == "autoformer":
        label_len = int(config.build["label_len"])
        decoder = torch.cat((x[:, -label_len:, :], torch.zeros_like(y[:, None, :])), dim=1)
        decoder_mark = torch.cat((x_mark[:, -label_len:, :], y_mark), dim=1)
        return model(x, x_mark, decoder, decoder_mark)[:, -1, :]
    return model(x, None, None, None)[:, -1, :]


def batch_loss(model_name: str, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if model_name == "autoformer":
        return F.mse_loss(prediction[:, 0], target[:, 0])
    # Match the official iTransformer training path: all forecast variates are
    # optimized together with one unweighted MSE.  The previous
    # capacity-dominant objective (primary + 0.2 * auxiliary) was a battery-
    # specific optimization and made the comparison less source-faithful.
    return F.mse_loss(prediction, target)


def move_batch(batch, device: torch.device):
    return tuple(value.to(device) for value in batch)


@torch.no_grad()
def validation_loss(model_name, model, loader, device, config) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        prediction = forward_model(model_name, model, batch, config)
        loss = batch_loss(model_name, prediction, batch[1])
        total += float(loss) * len(batch[1])
        count += len(batch[1])
    return total / max(count, 1)


def fit(model_name, model, train_set, validation_set, config, device, seed, tag):
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        generator=generator, num_workers=0,
    )
    validation_loader = DataLoader(
        validation_set, batch_size=config.batch_size, shuffle=False, num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.to(device)
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_model(model_name, model, batch, config)
            loss = batch_loss(model_name, prediction, batch[1])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            if config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            total += float(loss.detach()) * len(batch[1])
            count += len(batch[1])
        val = validation_loss(model_name, model, validation_loader, device, config)
        row = {"epoch": epoch, "train_mse": total / max(count, 1), "validation_mse": val}
        history.append(row)
        print(f"{tag} epoch={epoch:03d} train={row['train_mse']:.8f} val={val:.8f}", flush=True)
        if val < best_loss - 1e-8:
            best_state = copy.deepcopy(model.state_dict())
            best_loss = val
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_loss


@torch.no_grad()
def predict(model_name, model, dataset, config, device, minimum, maximum):
    loader = DataLoader(dataset, batch_size=max(config.batch_size, 256), shuffle=False, num_workers=0)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        batch = move_batch(batch, device)
        output = forward_model(model_name, model, batch, config)
        predictions.append(output[:, 0].cpu().numpy())
        targets.append(batch[1][:, 0].cpu().numpy())
    span = max(maximum - minimum, 1e-8)
    return np.concatenate(targets) * span + minimum, np.concatenate(predictions) * span + minimum


def run_fold(model_name, dataset, test_battery, seed, output_root, device, max_epochs, data_version):
    protocol = PROTOCOLS[dataset]
    output = output_root / model_name / dataset / test_battery / f"seed_{seed}"
    result_path = output / "results.json"
    config = model_config(model_name, dataset, protocol.seq_len)
    if max_epochs is not None:
        config = replace(config, max_epochs=max_epochs, patience=min(config.patience, max_epochs))
    # JSON-normalize tuples so resumability comparisons survive serialization.
    expected_config = json.loads(json.dumps(asdict(config)))
    if result_path.exists():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "complete"
                and existing.get("native_config") == expected_config
                and existing.get("data_version") == data_version
            ):
                print(f"skip complete: {output}", flush=True)
                return
        except (OSError, json.JSONDecodeError):
            pass

    frames = load_frames(dataset, data_version)
    train_names = [name for name in protocol.batteries if name != test_battery]
    train_samples, validation_samples = chronological_samples(frames, train_names, protocol.seq_len)
    raw = {name: frame["Capacity"].to_numpy(dtype=np.float32) for name, frame in frames.items()}
    train_capacity = np.concatenate([
        raw[name][: int(math.floor(len(raw[name]) * 0.8))] for name in train_names
    ])
    minimum, maximum = float(train_capacity.min()), float(train_capacity.max())
    span = max(maximum - minimum, 1e-8)
    feature_columns = {
        "capacity": 0,
        "delta": 1,
        "short_trend": 2,
        "long_trend": 3,
    }
    selected_columns = [feature_columns[name] for name in config.input_features]
    values = {
        name: degradation_features((capacity - minimum) / span)[:, selected_columns]
        for name, capacity in raw.items()
    }
    cycles = {name: frame["Cycle"].to_numpy(dtype=np.float32) for name, frame in frames.items()}
    train_set = BatteryWindows(model_name, values, cycles, train_samples, protocol.seq_len)
    validation_set = BatteryWindows(model_name, values, cycles, validation_samples, protocol.seq_len)
    test_samples = [(test_battery, target) for target in range(protocol.seq_len, len(raw[test_battery]))]
    test_set = BatteryWindows(model_name, values, cycles, test_samples, protocol.seq_len)

    seed_everything(seed)
    model = build_model(model_name, config)
    started = time.time()
    tag = f"[{model_name}/{dataset}/{test_battery}/seed={seed}]"
    model, history, best_epoch, best_loss = fit(
        model_name, model, train_set, validation_set, config, device, seed, tag,
    )
    y_true, y_pred = predict(model_name, model, test_set, config, device, minimum, maximum)
    test_cycles = frames[test_battery]["Cycle"].to_numpy(dtype=int)[protocol.seq_len:]
    metrics, prediction_rows = evaluate_predictions(test_cycles, y_true, y_pred, protocol)
    for row in prediction_rows:
        row.update({"model": model_name, "dataset": dataset, "battery": test_battery, "seed": seed})
    result = {
        "status": "complete",
        "model": model_name,
        "dataset": dataset,
        "test_battery": test_battery,
        "train_batteries": train_names,
        "seed": seed,
        "task": f"{protocol.seq_len}-cycle rolling one-step capacity prediction",
        "start_points": list(protocol.start_points),
        "metrics": metrics,
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "elapsed_seconds": time.time() - started,
        "normalization": {
            "type": "train-only capacity min-max",
            "minimum_ah": minimum,
            "maximum_ah": maximum,
        },
        "architecture_contract": (
            "progressive decomposition + FFT Auto-Correlation"
            if model_name == "autoformer"
            else "causal degradation variates as tokens + attention across variates"
        ),
        "native_config": expected_config,
        "data_version": data_version,
        "processed_summary": (
            f"data/{'version3' if data_version == 'version3' else 'processed-version2.0'}"
            if data_version in ("version2.0", "version3")
            else str(protocol.summary_path.relative_to(ROOT))
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "result": result}, output / "checkpoint.pt")
    write_csv(output / "training_history.csv", history)
    write_csv(output / "predictions.csv", prediction_rows)
    write_json(result_path, result)
    print(f"complete: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("autoformer", "itransformer"), required=True)
    parser.add_argument("--datasets", nargs="+", choices=tuple(PROTOCOLS), default=list(PROTOCOLS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--test-batteries", nargs="+", default=None)
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "outputs" / "comparison_models_aligned_10seeds",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-epochs", type=int, default=None, help="Smoke-test override only.")
    parser.add_argument(
        "--data-version", choices=("processed", "version2.0", "version3"),
        default="version3",
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    output_root = args.output_root.resolve()
    protocol_path = output_root / "protocol.json"
    if not protocol_path.exists():
        manifest = protocol_manifest()
        manifest.update({"runner": Path(__file__).name, "models": [args.model]})
        write_json(protocol_path, manifest)
    device = torch.device(args.device)
    for dataset in args.datasets:
        batteries = [PROTOCOLS[dataset].batteries[0]]
        if args.test_batteries:
            batteries = [name for name in PROTOCOLS[dataset].batteries if name in args.test_batteries]
            if not batteries:
                parser.error(f"none of --test-batteries belongs to {dataset}")
        for battery in batteries:
            for seed in args.seeds:
                run_fold(
                    args.model, dataset, battery, seed, output_root, device,
                    args.max_epochs, args.data_version,
                )


if __name__ == "__main__":
    main()
