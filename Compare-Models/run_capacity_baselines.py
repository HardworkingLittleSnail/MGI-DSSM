"""Unified processed-data benchmark for PatchFormer and RUL-Mamba."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "RUL-Mamba"))

from comparison_protocol import (  # noqa: E402
    DEFAULT_SEEDS, PROTOCOLS, chronological_samples, evaluate_predictions,
    load_summary, protocol_manifest, seed_everything, write_csv, write_json,
)
from Models.RULMamba import RULMamba  # noqa: E402

sys.path.insert(0, str(HERE / "PatchFormer"))
from ModelsModify.PatchFormer import PatchFormer  # noqa: E402


class CapacityWindows(Dataset):
    def __init__(self, values, samples, seq_len, minimum, maximum):
        self.values, self.samples, self.seq_len = values, samples, seq_len
        self.minimum, self.span = float(minimum), max(float(maximum - minimum), 1e-8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, target = self.samples[index]
        capacity = self.values[name]
        x = (capacity[target - self.seq_len:target] - self.minimum) / self.span
        y = (capacity[target] - self.minimum) / self.span
        return torch.tensor(x[:, None], dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


@dataclass(frozen=True)
class NativeConfig:
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    build: dict[str, object]


def native_config(model_name: str, dataset: str, seq_len: int) -> NativeConfig:
    if model_name == "patchformer":
        # Released implementation defaults. Table 4 fixes batch size/FAN units;
        # sequence length is replaced only by the common task contract.
        batch = 16 if dataset == "nasa" else 128
        return NativeConfig(1e-3, batch, 200, 10, {
            "patch_len": 2, "seq_len": seq_len, "pred_len": 1, "enc_in": 1,
            "d_model": 16, "factor": 3, "dropout": 0.1,
            "output_attention": False, "n_heads": 8, "activation": "gelu", "e_layers": 2,
        })
    # RUL-Mamba Table 4. CALCE is an out-of-paper transfer setting and uses
    # the NASA univariate configuration with the common 64-cycle window.
    if dataset == "nasa":
        build = {"enc_in": 1, "d_model": 48, "n_dec_layer": 1, "dropout": 0.0615, "expand": 2}
        lr = 0.0022
    elif dataset == "tju":
        build = {"enc_in": 1, "d_model": 16, "n_dec_layer": 2, "dropout": 0.1, "expand": 2}
        lr = 0.001
    else:
        build = {"enc_in": 1, "d_model": 48, "n_dec_layer": 1, "dropout": 0.0615, "expand": 2}
        lr = 0.0022
    return NativeConfig(lr, 16, 200, 20, build)


def smape(prediction, target):
    return (2.0 * (prediction - target).abs() / (prediction.abs() + target.abs() + 1e-8)).mean()


def forward_model(model_name, model, x):
    if model_name == "patchformer":
        return model(x).reshape(-1)
    return model(x_enc=x, x_dec=None).reshape(-1)


@torch.no_grad()
def validation_loss(model_name, model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = smape(forward_model(model_name, model, x), y)
        total += float(loss) * len(y)
        count += len(y)
    return total / max(count, 1)


def fit(model_name, model, train_set, validation_set, config, device, seed, log):
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_set, batch_size=config.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.to(device)
    best_state, best_loss, best_epoch, stale, history = None, float("inf"), 0, 0, []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_model(model_name, model, x)
            loss = smape(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)
        val = validation_loss(model_name, model, validation_loader, device)
        row = {"epoch": epoch, "train_smape": total / max(count, 1), "validation_smape": val}
        history.append(row)
        print(f"{log} epoch={epoch:03d} train={row['train_smape']:.7f} val={val:.7f}", flush=True)
        if val < best_loss - 1e-8:
            best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), val, epoch, 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_loss


@torch.no_grad()
def predict(model_name, model, dataset, batch_size, device, minimum, maximum):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    output, target = [], []
    model.eval()
    for x, y in loader:
        output.append(forward_model(model_name, model, x.to(device)).cpu().numpy())
        target.append(y.numpy())
    output = np.concatenate(output)
    target = np.concatenate(target)
    span = max(maximum - minimum, 1e-8)
    return target * span + minimum, output * span + minimum


def run_fold(model_name, dataset_name, battery, seed, output_root, device, max_epochs=None):
    protocol = PROTOCOLS[dataset_name]
    output = output_root / model_name / dataset_name / battery / f"seed_{seed}"
    done = output / "results.json"
    if done.exists():
        try:
            if json.loads(done.read_text(encoding="utf-8")).get("status") == "complete":
                print(f"skip complete: {output}", flush=True)
                return
        except (OSError, json.JSONDecodeError):
            pass
    frames = load_summary(protocol)
    values = {name: frame["Capacity"].to_numpy(dtype=np.float32) for name, frame in frames.items()}
    train_names = [name for name in protocol.batteries if name != battery]
    train_samples, validation_samples = chronological_samples(frames, train_names, protocol.seq_len)
    # Model normalization is estimated from the training portion only.
    train_values = np.concatenate([values[name][: int(len(values[name]) * 0.8)] for name in train_names])
    minimum, maximum = float(train_values.min()), float(train_values.max())
    train_set = CapacityWindows(values, train_samples, protocol.seq_len, minimum, maximum)
    validation_set = CapacityWindows(values, validation_samples, protocol.seq_len, minimum, maximum)
    test_samples = [(battery, target) for target in range(protocol.seq_len, len(values[battery]))]
    test_set = CapacityWindows(values, test_samples, protocol.seq_len, minimum, maximum)
    config = native_config(model_name, dataset_name, protocol.seq_len)
    if max_epochs is not None:
        config = NativeConfig(config.learning_rate, config.batch_size, max_epochs, min(config.patience, max_epochs), config.build)
    seed_everything(seed)
    model = PatchFormer(**config.build) if model_name == "patchformer" else RULMamba(**config.build)
    started = time.time()
    tag = f"[{model_name}/{dataset_name}/{battery}/seed={seed}]"
    model, history, best_epoch, best_loss = fit(model_name, model, train_set, validation_set, config, device, seed, tag)
    y_true, y_pred = predict(model_name, model, test_set, config.batch_size, device, minimum, maximum)
    cycles = frames[battery]["Cycle"].to_numpy(dtype=int)[protocol.seq_len:]
    metrics, prediction_rows = evaluate_predictions(cycles, y_true, y_pred, protocol)
    for row in prediction_rows:
        row.update({"model": model_name, "dataset": dataset_name, "battery": battery, "seed": seed})
    result = {
        "status": "complete", "model": model_name, "dataset": dataset_name,
        "test_battery": battery, "train_batteries": train_names, "seed": seed,
        "task": f"{protocol.seq_len}-cycle rolling one-step capacity prediction",
        "start_points": list(protocol.start_points), "metrics": metrics,
        "best_epoch": best_epoch, "best_validation_smape": best_loss,
        "elapsed_seconds": time.time() - started,
        "normalization": {"type": "train-only min-max", "minimum_ah": minimum, "maximum_ah": maximum},
        "native_config": {"learning_rate": config.learning_rate, "batch_size": config.batch_size,
                          "max_epochs": config.max_epochs, "patience": config.patience, "build": config.build},
        "processed_summary": str(protocol.summary_path.relative_to(ROOT)),
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "result": result}, output / "checkpoint.pt")
    write_csv(output / "training_history.csv", history)
    write_csv(output / "predictions.csv", prediction_rows)
    write_json(done, result)
    print(f"complete: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("patchformer", "rul-mamba"), required=True)
    parser.add_argument("--datasets", nargs="+", choices=tuple(PROTOCOLS), default=list(PROTOCOLS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--test-batteries", nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "comparison_models_10seeds")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-epochs", type=int, default=None, help="Smoke-test override only")
    args = parser.parse_args()
    device = torch.device(args.device)
    write_json(args.output_root / "protocol.json", protocol_manifest())
    for dataset in args.datasets:
        batteries = list(PROTOCOLS[dataset].batteries)
        if args.test_batteries:
            batteries = [name for name in batteries if name in args.test_batteries]
        for battery in batteries:
            for seed in args.seeds:
                run_fold(args.model, dataset, battery, seed, args.output_root, device, args.max_epochs)


if __name__ == "__main__":
    main()
