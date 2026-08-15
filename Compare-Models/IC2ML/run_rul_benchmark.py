from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, Dataset

from models.IC2ML import Model as OriginalModel
from models.IC2ML_direct import Model as DirectModel
from prepare_shared_native_inputs import build as build_native_inputs

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from comparison_protocol import PROTOCOLS


BATTERIES = ("B0005", "B0006", "B0007", "B0018")
DEFAULT_SEEDS = (7, 17, 27, 37, 47, 57, 67, 77, 87, 97)
START_POINTS_BY_BATTERY = {name: (50, 70, 90) for name in BATTERIES}
RATED_CAPACITY = 2.0
RUL_SCALE = 200.0
CAPACITY_SCALING = "ah"
INPUT_CAPACITY_LIKE = True
FEATURE_MINIMUM: np.ndarray | None = None
FEATURE_MAXIMUM: np.ndarray | None = None
RUNNER_VERSION = "direct-history-ridge-paper-multitask-v27"
TRAIN_RUL_EOL_FRACTION = 0.80


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def first_consecutive_crossing(values: np.ndarray, threshold: float) -> int:
    for index in range(len(values) - 1):
        if values[index] <= threshold and values[index + 1] <= threshold:
            return index - 1
    return len(values)


def capacity_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
    squared_error_sum = float(np.sum(np.square(y_true - y_pred)))
    total = float(np.sum(np.square(y_true - y_true.mean())))
    r2 = 1.0 - squared_error_sum / total if total > 0 else float("nan")
    true_re = first_consecutive_crossing(y_true, RATED_CAPACITY * 0.7)
    # Final MGI-DSSM 10-run protocol uses the symmetric consecutive-threshold
    # rule for both true and predicted curves (see its final summary report).
    pred_re = first_consecutive_crossing(y_pred, RATED_CAPACITY * 0.7)
    ae = abs(true_re - pred_re)
    re = min(ae / max(abs(true_re), 1), 1.0)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "RUL_real": int(true_re + 1),
        "RUL_pred": int(pred_re + 1),
        "AE": int(ae),
        "RE": float(re),
    }


def load_shared(
    root: Path, dataset: str, voltage_range: tuple[float, float],
    data_version: str = "processed",
):
    path = build_native_inputs(
        root.parents[1], dataset, voltage_range, data_version=data_version
    )
    return np.load(path, allow_pickle=True).item()


def load_calce(
    root: Path, voltage_range: tuple[float, float]
) -> dict[str, dict[str, np.ndarray]]:
    voltage_start, voltage_end = voltage_range
    path = (
        root
        / "data"
        / "CALCE data"
        / f"CALCE_IC2ML_charge_{voltage_start:g}-{voltage_end:g}.npy"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Aligned CALCE IC2ML features not found: {path}. "
            "Generate them from the raw charge curves with prepare_calce_ic2ml.py."
        )
    payload = np.load(path, allow_pickle=True)[0]
    return {
        str(name): {
            "increments": np.asarray(values["increments"], dtype=np.float32),
            "capacities": np.asarray(values["capacities"], dtype=np.float32),
            "cycles": np.asarray(values["cycles"], dtype=np.int64),
        }
        for name, values in payload.items()
    }


def normalize_capacity(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return values / RATED_CAPACITY if CAPACITY_SCALING == "rated" else values


def denormalize_capacity(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return values * RATED_CAPACITY if CAPACITY_SCALING == "rated" else values


def eol_index(capacities: np.ndarray) -> int:
    # IC2ML defines its auxiliary RUL at 80% SOH.  The shared benchmark's
    # threshold-derived evaluation RUL remains 70% and is computed separately.
    crossing = first_consecutive_crossing(
        capacities, RATED_CAPACITY * TRAIN_RUL_EOL_FRACTION
    )
    return max(0, min(crossing, len(capacities) - 1))


class Windows(Dataset):
    def __init__(
        self,
        data: dict[str, dict[str, np.ndarray]],
        names: list[str],
        bounds: tuple[float, float],
        seq_len: int,
        samples: list[tuple[str, int]] | None = None,
        include_cycle_input: bool = False,
    ) -> None:
        self.data = data
        self.bounds = bounds
        self.seq_len = seq_len
        self.include_cycle_input = include_cycle_input
        # EOL is constant for a battery. Computing it here avoids rescanning an
        # approximately 900-cycle CALCE trajectory for every sample/epoch.
        self.eol_by_name = {
            name: eol_index(data[name]["capacities"]) for name in names
        }
        self.samples = samples or [
            (name, target)
            for name in names
            for target in range(seq_len, len(data[name]["capacities"]))
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        name, target = self.samples[index]
        item = self.data[name]
        start = target - self.seq_len
        # Official files store capacity increments in mAh and divide by 1000.
        # Our aligned native-input cache already stores the same quantity in Ah.
        x = item["increments"][start:target]
        if CAPACITY_SCALING == "rated" and INPUT_CAPACITY_LIKE:
            x = x / RATED_CAPACITY
        if FEATURE_MINIMUM is not None and FEATURE_MAXIMUM is not None:
            span = np.where(FEATURE_MAXIMUM > FEATURE_MINIMUM,
                            FEATURE_MAXIMUM - FEATURE_MINIMUM, 1.0)
            x = (x - FEATURE_MINIMUM) / span
        history = normalize_capacity(item["capacities"][start:target], self.bounds)
        future = normalize_capacity(item["capacities"][target:target + 1], self.bounds)
        # Fixed dataset-level scaling preserves the RUL target while keeping
        # its auxiliary MSE commensurate with normalized SOH/capacity losses.
        remaining = max(0, self.eol_by_name[name] - (target - 1)) / RUL_SCALE
        x_tensor = torch.tensor(x, dtype=torch.float32)
        cycle_tensor = torch.tensor(
            item["cycles"][start:target] / RUL_SCALE, dtype=torch.float32
        )
        history_tensor = torch.tensor(history, dtype=torch.float32)
        model_input = (
            (x_tensor, cycle_tensor, history_tensor)
            if self.include_cycle_input else x_tensor
        )
        return model_input, (
            torch.tensor(future, dtype=torch.float32),
            history_tensor,
            torch.tensor(remaining, dtype=torch.float32),
        )


def split_train_validation(
    data: dict[str, dict[str, np.ndarray]], names: list[str], seq_len: int,
    fraction: float, mode: str, requested_validation: str | None,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[str], list[str]]:
    if mode == "cell":
        if len(names) < 2:
            raise ValueError("cell-level validation requires at least two non-test cells")
        validation_name = requested_validation or ("B0007" if "B0007" in names else names[-1])
        if validation_name not in names:
            raise ValueError(f"validation battery {validation_name!r} is not a non-test cell")
        train_names = [name for name in names if name != validation_name]
        train = [
            (name, target) for name in train_names
            for target in range(seq_len, len(data[name]["capacities"]))
        ]
        validation = [
            (validation_name, target)
            for target in range(seq_len, len(data[validation_name]["capacities"]))
        ]
        return train, validation, train_names, [validation_name]

    # Preserve leave-one-cell-out testing while using every remaining cell for
    # fitting.  This is the common benchmark split: the chronological tail of
    # each training cell is validation data and no target-cell sample is seen.
    train: list[tuple[str, int]] = []
    validation: list[tuple[str, int]] = []
    for name in names:
        length = len(data[name]["capacities"])
        split = int(math.floor(length * (1.0 - fraction)))
        if split <= seq_len or length - split < 1:
            raise ValueError(
                f"{name} is too short for seq_len={seq_len} and validation fraction={fraction}"
            )
        train.extend((name, target) for target in range(seq_len, split))
        validation.extend((name, target) for target in range(split, length))
    return train, validation, names, names


@dataclass
class TrainResult:
    model: torch.nn.Module
    best_epoch: int
    best_validation_mae_ah: float
    best_validation_objective: float
    history: list[dict[str, float | int]]


def train_fold(
    data: dict[str, dict[str, np.ndarray]],
    test_name: str,
    seed: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> TrainResult:
    global FEATURE_MINIMUM, FEATURE_MAXIMUM
    set_seed(seed)
    candidate_names = [name for name in BATTERIES if name != test_name]
    bounds = (0.0, 1.0)
    train_samples, validation_samples, train_names, validation_names = split_train_validation(
        data, candidate_names, args.seq_len, args.validation_fraction, args.validation_mode,
        args.validation_battery,
    )
    if INPUT_CAPACITY_LIKE:
        FEATURE_MINIMUM = FEATURE_MAXIMUM = None
    else:
        training_features = np.concatenate(
            [data[name]["increments"] for name in train_names], axis=0
        ).astype(np.float64)
        FEATURE_MINIMUM = np.nanmin(training_features, axis=0)
        FEATURE_MAXIMUM = np.nanmax(training_features, axis=0)
    direct = args.model_variant == "direct"
    include_auxiliary_input = direct or args.use_capacity_history
    train_data = Windows(
        data, train_names, bounds, args.seq_len, train_samples, include_auxiliary_input
    )
    validation_data = Windows(
        data, validation_names, bounds, args.seq_len, validation_samples,
        include_auxiliary_input,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = DirectModel if direct else OriginalModel
    model = model_class(SimpleNamespace(
        context=args.seq_len, horizon=1, hidden_dim=args.hidden_dim,
        input_dim=args.input_dim, use_cycle_input=args.use_cycle_input,
        use_capacity_history=args.use_capacity_history,
    )).to(device)
    if args.initialize_history_readout_ridge:
        if not direct or not args.use_capacity_history:
            raise ValueError("History Ridge initialization requires the adapted direct model.")
        histories = np.stack([
            normalize_capacity(
                data[name]["capacities"][target - args.seq_len:target], bounds
            )
            for name, target in train_samples
        ])
        futures = np.asarray([
            normalize_capacity(
                data[name]["capacities"][target:target + 1], bounds
            )[0]
            for name, target in train_samples
        ])
        ridge = Ridge(alpha=1e-6).fit(histories, futures)
        with torch.no_grad():
            model.trajectory_predictor.weight.zero_()
            model.trajectory_predictor.weight[0, -args.seq_len:] = torch.as_tensor(
                ridge.coef_, dtype=model.trajectory_predictor.weight.dtype,
                device=device,
            )
            model.trajectory_predictor.bias.fill_(float(ridge.intercept_))
    # The released IC2ML trainer uses Adam (not AdamW).
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.capacity_loss == "mae":
        capacity_criterion = torch.nn.L1Loss()
    elif args.capacity_loss == "smooth_l1":
        capacity_criterion = torch.nn.SmoothL1Loss(beta=args.smooth_l1_beta)
    else:
        capacity_criterion = torch.nn.MSELoss()
    rul_criterion = torch.nn.MSELoss()
    best_score, best_state, best_epoch, best_mae, stale = (
        float("inf"), None, 0, float("inf"), 0
    )

    history_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for inputs, targets in train_loader:
            future, history, rul = (value.to(device) for value in targets)
            optimizer.zero_grad(set_to_none=True)
            if direct:
                increments, cycles, observed_history = (value.to(device) for value in inputs)
                pred_history, pred_future, pred_rul = model(
                    increments, cycles, observed_history
                )
            elif args.use_capacity_history:
                increments, _, observed_history = (value.to(device) for value in inputs)
                pred_history, pred_future, pred_rul = model(
                    increments, observed_capacity_history=observed_history
                )
            else:
                pred_history, pred_future, pred_rul = model(inputs.to(device))
            loss = args.history_loss_weight * capacity_criterion(pred_history, history)
            loss = loss + args.trajectory_loss_weight * capacity_criterion(pred_future, future)
            loss = loss + args.rul_loss_weight * rul_criterion(pred_rul, rul)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        predictions, truths, validation_losses = [], [], []
        with torch.no_grad():
            for inputs, targets in validation_loader:
                future, history, rul = (value.to(device) for value in targets)
                if direct:
                    increments, cycles, observed_history = (value.to(device) for value in inputs)
                    pred_history, pred_future, pred_rul = model(
                        increments, cycles, observed_history
                    )
                elif args.use_capacity_history:
                    increments, _, observed_history = (value.to(device) for value in inputs)
                    pred_history, pred_future, pred_rul = model(
                        increments, observed_capacity_history=observed_history
                    )
                else:
                    pred_history, pred_future, pred_rul = model(inputs.to(device))
                validation_loss = args.history_loss_weight * capacity_criterion(pred_history, history)
                validation_loss = validation_loss + args.trajectory_loss_weight * capacity_criterion(pred_future, future)
                validation_loss = validation_loss + args.rul_loss_weight * rul_criterion(pred_rul, rul)
                validation_losses.append(float(validation_loss.item()))
                pred = pred_future.squeeze(-1).cpu().numpy()
                predictions.append(pred)
                truths.append(targets[0].squeeze(-1).numpy())
        pred_ah = denormalize_capacity(np.concatenate(predictions), bounds)
        true_ah = denormalize_capacity(np.concatenate(truths), bounds)
        validation_mae = float(np.mean(np.abs(pred_ah - true_ah)))
        score = (
            float(np.mean(validation_losses))
            if args.selection_objective == "multitask"
            else validation_mae
        )
        history_rows.append({"epoch": epoch, "train_multitask_mse": float(np.mean(train_losses)),
                             "validation_multitask_mse": float(np.mean(validation_losses)),
                             "validation_mae_ah": validation_mae})
        print(f"[ic2ml/{args.dataset}/{test_name}/seed={seed}] epoch={epoch:03d} "
              f"train={np.mean(train_losses):.7f} val={score:.7f}", flush=True)
        if score < best_score - 1e-8:
            best_score, best_epoch, best_mae, stale = score, epoch, validation_mae, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")

    # After validation has fixed the training horizon, refit from scratch on
    # every non-test battery. The held-out test battery is never used here.
    if args.refit_all:
        set_seed(seed)
        model = model_class(SimpleNamespace(
            context=args.seq_len, horizon=1, hidden_dim=args.hidden_dim,
            input_dim=args.input_dim, use_cycle_input=args.use_cycle_input,
            use_capacity_history=args.use_capacity_history,
        )).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        all_samples = [
            (name, target) for name in candidate_names
            for target in range(args.seq_len, len(data[name]["capacities"]))
        ]
        refit_data = Windows(
            data, candidate_names, bounds, args.seq_len, all_samples,
            direct or args.use_capacity_history,
        )
        refit_generator = torch.Generator().manual_seed(seed)
        refit_loader = DataLoader(
            refit_data, batch_size=args.batch_size, shuffle=True,
            generator=refit_generator,
        )
        for refit_epoch in range(1, best_epoch + 1):
            model.train()
            refit_losses = []
            for inputs, targets in refit_loader:
                future, history, rul = (value.to(device) for value in targets)
                optimizer.zero_grad(set_to_none=True)
                if direct:
                    increments, cycles, observed_history = (value.to(device) for value in inputs)
                    pred_history, pred_future, pred_rul = model(
                        increments, cycles, observed_history
                    )
                elif args.use_capacity_history:
                    increments, _, observed_history = (value.to(device) for value in inputs)
                    pred_history, pred_future, pred_rul = model(
                        increments, observed_capacity_history=observed_history
                    )
                else:
                    pred_history, pred_future, pred_rul = model(inputs.to(device))
                loss = args.history_loss_weight * capacity_criterion(pred_history, history)
                loss = loss + args.trajectory_loss_weight * capacity_criterion(pred_future, future)
                loss = loss + args.rul_loss_weight * rul_criterion(pred_rul, rul)
                loss.backward()
                optimizer.step()
                refit_losses.append(float(loss.detach().cpu()))
            print(
                f"[ic2ml/{args.dataset}/{test_name}/seed={seed}] "
                f"refit={refit_epoch:03d}/{best_epoch:03d} "
                f"train={np.mean(refit_losses):.7f}", flush=True,
            )
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        train_names = candidate_names
    else:
        model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "bounds": bounds,
        "test_battery": test_name,
        "train_batteries": train_names,
        "validation_batteries": validation_names,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_mae_ah": best_mae,
        "best_validation_objective": best_score,
        "args": vars(args),
        "feature_minimum": FEATURE_MINIMUM,
        "feature_maximum": FEATURE_MAXIMUM,
    }, output_dir / "checkpoint.pth")
    return TrainResult(model, best_epoch, best_mae, best_score, history_rows)


@torch.no_grad()
def evaluate_start(
    model: torch.nn.Module,
    data: dict[str, dict[str, np.ndarray]],
    test_name: str,
    bounds: tuple[float, float],
    seq_len: int,
    start_point: int,
    batch_size: int,
    model_variant: str,
    use_capacity_history: bool = False,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    item = data[test_name]
    samples = [
        (test_name, target)
        for target in range(seq_len, len(item["capacities"]))
        if int(item["cycles"][target]) >= start_point
    ]
    direct = model_variant == "direct"
    loader = DataLoader(
        Windows(
            data, [test_name], bounds, seq_len, samples,
            direct or use_capacity_history,
        ), batch_size=batch_size
    )
    device = next(model.parameters()).device
    predictions, truths, predicted_rul, true_rul = [], [], [], []
    for inputs, targets in loader:
        if direct:
            increments, cycles, observed_history = (value.to(device) for value in inputs)
            outputs = model(increments, cycles, observed_history)
        elif use_capacity_history:
            increments, _, observed_history = (value.to(device) for value in inputs)
            outputs = model(
                increments, observed_capacity_history=observed_history
            )
        else:
            outputs = model(inputs.to(device))
        predictions.append(outputs[1].cpu().squeeze(-1).numpy())
        truths.append(targets[0].squeeze(-1).numpy())
        predicted_rul.append(outputs[2].cpu().numpy())
        true_rul.append(targets[2].numpy())
    pred_ah = denormalize_capacity(np.concatenate(predictions), bounds)
    true_ah = denormalize_capacity(np.concatenate(truths), bounds)
    metric = capacity_metrics(true_ah, pred_ah)
    direct_pred = np.concatenate(predicted_rul).astype(np.float64) * RUL_SCALE
    direct_true = np.concatenate(true_rul).astype(np.float64) * RUL_SCALE
    metric["direct_rul_mae_cycles"] = float(np.mean(np.abs(direct_true - direct_pred)))
    metric["direct_rul_rmse_cycles"] = float(
        np.sqrt(np.mean(np.square(direct_true - direct_pred)))
    )
    metric.update({"test_battery": test_name, "start_point": start_point, "num_windows": len(samples)})
    rows = [
        {
            "battery": test_name,
            "cycle": int(item["cycles"][target]),
            "start_point": start_point,
            "capacity_true_ah": float(true),
            "capacity_pred_ah": float(pred),
            "direct_rul_true_cycles": float(rul_true),
            "direct_rul_pred_cycles": float(rul_pred),
        }
        for (name, target), true, pred, rul_true, rul_pred in zip(
            samples, true_ah, pred_ah, direct_true, direct_pred
        )
    ]
    return metric, rows


def run_one(data, test_name: str, seed: int, args: argparse.Namespace, root: Path) -> dict:
    output_dir = root / args.output_root / args.dataset / test_name / f"seed_{seed}"
    result_path = output_dir / "results.json"
    expected_processed_summary = (
        f"data/{'version3' if args.data_version == 'version3' else 'processed-version2.0'}"
        if args.data_version in ("version2.0", "version3")
        else str(PROTOCOLS[args.dataset].summary_path.relative_to(ROOT))
    )
    if result_path.exists():
        try:
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            cached_config = cached.get("config", {})
            if (cached.get("status") == "complete"
                    and cached.get("implementation_version") == RUNNER_VERSION
                    and int(cached_config.get("seq_len", -1)) == int(args.seq_len)
                    and int(cached_config.get("epochs", -1)) == int(args.epochs)
                    and float(cached_config.get("voltage_start", float("nan"))) == float(args.voltage_start)
                    and float(cached_config.get("voltage_end", float("nan"))) == float(args.voltage_end)
                    and cached_config.get("selection_objective") == args.selection_objective
                    and cached_config.get("validation_mode") == args.validation_mode
                    and cached_config.get("capacity_scaling") == args.capacity_scaling
                    and cached_config.get("model_variant") == args.model_variant
                    and cached_config.get("data_version") == args.data_version
                    and cached.get("processed_summary") == expected_processed_summary
                    and int(cached_config.get("input_dim", -1)) == int(args.input_dim)
                    and cached_config.get("validation_battery") == args.validation_battery
                    and bool(cached_config.get("refit_all", False)) == bool(args.refit_all)
                    and float(cached_config.get("history_loss_weight", float("nan"))) == float(args.history_loss_weight)
                    and float(cached_config.get("trajectory_loss_weight", float("nan"))) == float(args.trajectory_loss_weight)
                    and float(cached_config.get("rul_loss_weight", float("nan"))) == float(args.rul_loss_weight)
                    and cached_config.get("capacity_loss") == args.capacity_loss
                    and bool(cached_config.get("use_cycle_input", False)) == bool(args.use_cycle_input)
                    and bool(cached_config.get("use_capacity_history", False)) == bool(args.use_capacity_history)
                    and bool(cached_config.get("initialize_history_readout_ridge", False)) == bool(args.initialize_history_readout_ridge)
                    and float(cached_config.get("smooth_l1_beta", float("nan"))) == float(args.smooth_l1_beta)):
                print(f"skip complete: {output_dir}", flush=True)
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({
        "status": "running", "model": "ic2ml", "dataset": args.dataset,
        "test_battery": test_name, "seed": seed,
        "implementation_version": RUNNER_VERSION,
    }, indent=2), encoding="utf-8")
    result = train_fold(data, test_name, seed, args, output_dir)
    checkpoint = torch.load(output_dir / "checkpoint.pth", map_location="cpu", weights_only=False)
    bounds = tuple(checkpoint["bounds"])
    folds, prediction_rows = [], []
    for start in START_POINTS_BY_BATTERY[test_name]:
        metric, rows = evaluate_start(
            result.model, data, test_name, bounds, args.seq_len, start, args.batch_size,
            args.model_variant, args.use_capacity_history
        )
        folds.append(metric)
        prediction_rows.extend(rows)
    payload = {
        "status": "complete",
        "model": "ic2ml",
        "implementation_version": RUNNER_VERSION,
        "dataset": args.dataset,
        "protocol": "MGI-DSSM-aligned IC2ML RUL evaluation",
        "task": f"past_{args.seq_len}_cycles_to_next_capacity_and_threshold_RUL",
        "trajectory_mode": (
            "non-anchored multimodal direct one-step SOH"
            if args.model_variant == "direct"
            else "paper multitask absolute capacity in Ah"
        ),
        "training_rul_eol_fraction": TRAIN_RUL_EOL_FRACTION,
        "seed": seed,
        "test_battery": test_name,
        "best_epoch": result.best_epoch,
        "best_validation_mae_ah": result.best_validation_mae_ah,
        "best_validation_objective": result.best_validation_objective,
        "folds": folds,
        "config": vars(args),
        "processed_summary": expected_processed_summary,
        "runtime": {
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.history[0]))
        writer.writeheader()
        writer.writerows(result.history)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    result_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"{args.dataset.upper()} {test_name} seed={seed} epoch={result.best_epoch} "
        f"MAE={np.mean([f['mae'] for f in folds]):.6f}Ah "
        f"RMSE={np.mean([f['rmse'] for f in folds]):.6f}Ah",
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="MGI-DSSM-aligned IC2ML RUL benchmark")
    parser.add_argument("--dataset", choices=("nasa", "calce", "tju"), default="nasa")
    parser.add_argument("--model-variant", choices=("original", "direct"), default="direct")
    parser.add_argument(
        "--data-version", choices=("processed", "version2.0", "version3"), default="version3"
    )
    parser.add_argument("--output-root", default="../../outputs/comparison_models_native_10seeds/ic2ml")
    parser.add_argument("--test-batteries", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--seq-len", type=int, default=None)
    # The original repository exposes a much larger ceiling, but the shared
    # small-data benchmark uses a fixed 100-epoch budget for tractable runs.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--use-cycle-input", action="store_true")
    parser.add_argument("--use-capacity-history", action="store_true")
    parser.add_argument("--initialize-history-readout-ridge", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--history-loss-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-loss-weight", type=float, default=1.0)
    parser.add_argument("--rul-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--capacity-loss", choices=("mse", "mae", "smooth_l1"), default="mse"
    )
    parser.add_argument("--smooth-l1-beta", type=float, default=0.01)
    parser.add_argument(
        "--refit-all", action="store_true",
        help="Refit for the selected epoch count on all non-test batteries.",
    )
    parser.add_argument("--capacity-scaling", choices=("ah", "rated"), default="rated")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--validation-mode", choices=("cell", "chronological"), default="cell",
        help="Use the author's battery/file-level validation or a chronological tail split.",
    )
    parser.add_argument("--validation-battery", default=None)
    parser.add_argument(
        "--selection-objective",
        choices=("multitask", "capacity_mae"),
        default="multitask",
    )
    parser.add_argument("--patience", type=int, default=100)
    # Paper Section 4.3 defines 3.6-3.7 V as the base IC2ML input interval.
    parser.add_argument("--voltage-start", type=float, default=3.6)
    parser.add_argument("--voltage-end", type=float, default=3.7)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    global BATTERIES, START_POINTS_BY_BATTERY, RATED_CAPACITY, RUL_SCALE
    global CAPACITY_SCALING, INPUT_CAPACITY_LIKE
    protocol = PROTOCOLS[args.dataset]
    BATTERIES = protocol.batteries
    START_POINTS_BY_BATTERY = {name: protocol.start_points for name in BATTERIES}
    RATED_CAPACITY = protocol.rated_capacity
    CAPACITY_SCALING = args.capacity_scaling
    INPUT_CAPACITY_LIKE = not (
        args.dataset == "tju" and args.data_version in ("version2.0", "version3")
    )
    # The released IC2ML loader uses current_rul / 100 for every dataset.
    RUL_SCALE = 100.0
    args.rul_target_unit = "cycles"
    args.rul_scale_cycles = RUL_SCALE
    args.seq_len = args.seq_len or protocol.seq_len
    data = load_shared(
        root, args.dataset, (args.voltage_start, args.voltage_end), args.data_version
    )
    args.input_dim = int(next(iter(data.values()))["increments"].shape[1])
    args.feature_source = (
        f"16 cycle-level non-capacity descriptors from {args.data_version}"
        if args.dataset == "tju" and args.data_version in ("version2.0", "version3")
        else (
            f"10-point incremental charge-capacity curve over "
            f"{args.voltage_start:g}-{args.voltage_end:g} V from {args.data_version}"
        )
    )
    if args.test_batteries is None:
        # Formal evaluation uses only the first cell; other cells form the
        # training pool and are never silently added to the result table.
        args.test_batteries = [BATTERIES[0]]
    invalid = sorted(set(args.test_batteries).difference(BATTERIES))
    if invalid:
        parser.error(f"invalid {args.dataset} batteries: {invalid}")
    for name in args.test_batteries:
        for seed in args.seeds:
            run_one(data, name, seed, args, root)


if __name__ == "__main__":
    main()
