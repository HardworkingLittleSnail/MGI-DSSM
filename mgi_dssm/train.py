from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import (
    FEATURE_COLUMNS,
    KINETIC_FEATURES,
    THERMO_FEATURES,
    PatchFormerWindowDataset,
    WindowStandardizer,
    battery_names,
    denormalize_capacity_norm,
    feature_indices,
    linear_extrapolation,
    prepare_patchformer_frame,
)
from .metrics import patchformer_capacity_metrics
from .model import MGIDSSMLite


@dataclass
class TrainConfig:
    seq_len: int = 64
    max_seq_len: int = 1000
    rated_capacity: float = 1.1
    start_points: Tuple[int, ...] = (65,)
    epochs: int = 50
    batch_size: int = 256
    lr: float = 3e-4
    hidden_dim: int = 64
    seed: int = 7
    recon_weight: float = 0.05
    monotone_weight: float = 0.01
    residual_anchor_weight: float = 0.02
    huber_beta: float = 0.005
    weight_decay: float = 1e-4
    dropout: float = 0.1


@dataclass
class FoldMetric:
    fold: str
    test_battery: str
    start_point: int
    train_batteries: List[str]
    num_windows: int
    residual_scale: float
    mae: float
    rmse: float
    r2: float
    RUL_real: int
    RUL_pred: int
    AE: int
    RE: float
    persistence_mae: float
    linear_last5_mae: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def _make_model(train_dataset: PatchFormerWindowDataset, config: TrainConfig) -> MGIDSSMLite:
    thermo_idx = feature_indices(FEATURE_COLUMNS, THERMO_FEATURES)
    kinetic_idx = feature_indices(FEATURE_COLUMNS, KINETIC_FEATURES)
    if len(train_dataset) == 0:
        initial = 0.9
    else:
        values = [float(train_dataset.groups[b].iloc[pos]["capacity_norm"]) for b, pos in train_dataset.samples]
        initial = float(np.nanmedian(values)) if values else 0.9
    return MGIDSSMLite(
        feature_dim=len(FEATURE_COLUMNS),
        thermo_indices=thermo_idx,
        kinetic_indices=kinetic_idx,
        hidden_dim=config.hidden_dim,
        initial_capacity_norm=initial,
        dropout=config.dropout,
    )


def _monotone_state_loss(states: torch.Tensor) -> torch.Tensor:
    if states.shape[1] < 2:
        return states.new_tensor(0.0)
    diffs = states[:, 1:, :5] - states[:, :-1, :5]
    return F.relu(-diffs).mean()


def train_one_fold(
    frame: pd.DataFrame,
    test_battery: str,
    config: TrainConfig,
    output_dir: Path | None = None,
) -> Tuple[List[FoldMetric], List[Dict[str, object]]]:
    set_seed(config.seed)
    names = battery_names(frame)
    train_batteries = [name for name in names if name != test_battery]
    prepared, bounds = prepare_patchformer_frame(
        frame,
        train_batteries=train_batteries,
        rated_capacity=config.rated_capacity,
        max_seq_len=config.max_seq_len,
    )
    scaler = WindowStandardizer.fit(prepared, train_batteries)
    train_dataset = PatchFormerWindowDataset(
        prepared,
        batteries=train_batteries,
        scaler=scaler,
        seq_len=config.seq_len,
        max_seq_len=config.max_seq_len,
    )
    if len(train_dataset) == 0:
        raise ValueError(f"No training windows for fold {test_battery}.")

    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    device = torch.device("cpu")
    model = _make_model(train_dataset, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    model.train()
    for _ in range(config.epochs):
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_hist = batch["y_hist"].to(device)
            outputs = model(x, last_capacity=y_hist[:, -1])
            pred = outputs["capacity"]
            hist_pred = outputs["history_capacity"]
            states = outputs["states"]

            residual = pred - y_hist[:, -1]
            loss = F.smooth_l1_loss(pred, y, beta=config.huber_beta)
            loss = loss + config.recon_weight * F.smooth_l1_loss(hist_pred, y_hist, beta=config.huber_beta)
            loss = loss + config.monotone_weight * _monotone_state_loss(states)
            loss = loss + config.residual_anchor_weight * residual.abs().mean()
            loss = loss + 0.001 * outputs["reversible_delta"].pow(2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    residual_scale = calibrate_residual_scale(model, train_dataset, config, device)
    fold_metrics: List[FoldMetric] = []
    prediction_rows: List[Dict[str, object]] = []
    for start_point in config.start_points:
        metric, rows = evaluate_fold(
            model=model,
            prepared=prepared,
            raw_frame=frame,
            bounds=bounds,
            scaler=scaler,
            test_battery=test_battery,
            train_batteries=train_batteries,
            start_point=int(start_point),
            config=config,
            residual_scale=residual_scale,
        )
        fold_metrics.append(metric)
        prediction_rows.extend(rows)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "feature_columns": FEATURE_COLUMNS,
                "scaler_mean": scaler.mean,
                "scaler_std": scaler.std,
                "capacity_bounds": bounds,
                "config": asdict(config),
                "test_battery": test_battery,
                "train_batteries": train_batteries,
                "residual_scale": residual_scale,
            },
            output_dir / f"{test_battery}_fold.pt",
        )
    return fold_metrics, prediction_rows


@torch.no_grad()
def calibrate_residual_scale(
    model: MGIDSSMLite,
    dataset: PatchFormerWindowDataset,
    config: TrainConfig,
    device: torch.device,
) -> float:
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    raw_residuals: List[np.ndarray] = []
    target_residuals: List[np.ndarray] = []
    model.eval()
    for batch in loader:
        x = batch["x"].to(device)
        y_hist = batch["y_hist"].to(device)
        out = model(x, last_capacity=y_hist[:, -1])
        raw_residuals.append((out["capacity"].cpu().numpy() - batch["y_hist"][:, -1].numpy()))
        target_residuals.append((batch["y"].numpy() - batch["y_hist"][:, -1].numpy()))
    if not raw_residuals:
        return 0.0
    raw = np.concatenate(raw_residuals)
    target = np.concatenate(target_residuals)
    grid = np.linspace(0.0, 1.0, 101)
    maes = [float(np.mean(np.abs(alpha * raw - target))) for alpha in grid]
    return float(grid[int(np.argmin(maes))])


@torch.no_grad()
def evaluate_fold(
    model: MGIDSSMLite,
    prepared: pd.DataFrame,
    raw_frame: pd.DataFrame,
    bounds: Tuple[float, float],
    scaler: WindowStandardizer,
    test_battery: str,
    train_batteries: List[str],
    start_point: int,
    config: TrainConfig,
    residual_scale: float,
) -> Tuple[FoldMetric, List[Dict[str, object]]]:
    dataset = PatchFormerWindowDataset(
        prepared,
        batteries=[test_battery],
        scaler=scaler,
        seq_len=config.seq_len,
        start_point=start_point,
        max_seq_len=config.max_seq_len,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    model.eval()

    rows: List[Dict[str, object]] = []
    y_true_all: List[float] = []
    y_pred_all: List[float] = []
    persistence_all: List[float] = []
    linear_all: List[float] = []

    raw_group = raw_frame[raw_frame["BatteryName"] == test_battery].sort_values("Cycle").reset_index(drop=True)
    by_cycle = {int(row["Cycle"]): row for _, row in raw_group.iterrows()}

    for batch in loader:
        last_norm_tensor = batch["y_hist"][:, -1]
        outputs = model(batch["x"], last_capacity=last_norm_tensor)
        raw_pred_norm = outputs["capacity"].cpu().numpy()
        last_norm = last_norm_tensor.cpu().numpy()
        pred_norm = last_norm + float(residual_scale) * (raw_pred_norm - last_norm)
        target_norm = batch["y"].cpu().numpy()
        pred_ah = denormalize_capacity_norm(pred_norm, bounds, config.rated_capacity)
        true_ah = denormalize_capacity_norm(target_norm, bounds, config.rated_capacity)
        cycles = batch["cycle"].cpu().numpy()
        states = outputs["next_state"].cpu().numpy()

        for idx, cycle in enumerate(cycles):
            cycle_i = int(cycle)
            hist = raw_group[raw_group["Cycle"].between(cycle_i - config.seq_len, cycle_i - 1)]
            hist_caps = hist["Capacity"].to_numpy(dtype=np.float32)
            persistence = float(hist_caps[-1]) if len(hist_caps) else float("nan")
            linear = linear_extrapolation(hist_caps, window=5) if len(hist_caps) else float("nan")
            true_capacity = float(by_cycle[cycle_i]["Capacity"]) if cycle_i in by_cycle else float(true_ah[idx])
            pred_capacity = float(pred_ah[idx])

            y_true_all.append(true_capacity)
            y_pred_all.append(pred_capacity)
            persistence_all.append(persistence)
            linear_all.append(linear)
            rows.append(
                {
                    "fold": f"leave_{test_battery}_out",
                    "test_battery": test_battery,
                    "start_point": int(start_point),
                    "cycle": cycle_i,
                    "capacity_true": true_capacity,
                    "capacity_pred": pred_capacity,
                    "capacity_pred_norm": float(pred_norm[idx]),
                    "capacity_pred_raw_norm": float(raw_pred_norm[idx]),
                    "residual_scale": float(residual_scale),
                    "persistence_pred": persistence,
                    "linear_last5_pred": linear,
                    "abs_error": abs(pred_capacity - true_capacity),
                    "persistence_abs_error": abs(persistence - true_capacity),
                    "linear_last5_abs_error": abs(linear - true_capacity),
                    **{f"z_{j}": float(states[idx, j]) for j in range(states.shape[1])},
                }
            )

    y_true = np.asarray(y_true_all, dtype=np.float64)
    y_pred = np.asarray(y_pred_all, dtype=np.float64)
    pers = np.asarray(persistence_all, dtype=np.float64)
    linear = np.asarray(linear_all, dtype=np.float64)
    if len(y_true) == 0:
        mae = rmse = persistence_mae = linear_mae = float("nan")
        r2 = RE = float("nan")
        RUL_real = RUL_pred = AE = 0
    else:
        main_metrics = patchformer_capacity_metrics(y_true, y_pred, rated_capacity=config.rated_capacity)
        mae = float(main_metrics["mae"])
        rmse = float(main_metrics["rmse"])
        r2 = float(main_metrics["r2"])
        RUL_real = int(main_metrics["rul_real"])
        RUL_pred = int(main_metrics["rul_pred"])
        AE = int(main_metrics["ae"])
        RE = float(main_metrics["re"])
        persistence_mae = float(np.mean(np.abs(pers - y_true)))
        linear_mae = float(np.mean(np.abs(linear - y_true)))

    return (
        FoldMetric(
            fold=f"leave_{test_battery}_out",
            test_battery=test_battery,
            start_point=int(start_point),
            train_batteries=train_batteries,
            num_windows=len(rows),
            residual_scale=float(residual_scale),
            mae=mae,
            rmse=rmse,
            r2=r2,
            RUL_real=RUL_real,
            RUL_pred=RUL_pred,
            AE=AE,
            RE=RE,
            persistence_mae=persistence_mae,
            linear_last5_mae=linear_mae,
        ),
        rows,
    )


def run_leave_one_battery_out(
    frame: pd.DataFrame,
    config: TrainConfig,
    output_dir: Path,
    test_batteries: Iterable[str] | None = None,
) -> Dict[str, object]:
    selected = list(test_batteries) if test_batteries else battery_names(frame)
    metrics: List[FoldMetric] = []
    predictions: List[Dict[str, object]] = []
    for test_battery in selected:
        fold_metrics, fold_predictions = train_one_fold(
            frame,
            test_battery,
            config,
            output_dir=output_dir / "checkpoints",
        )
        metrics.extend(fold_metrics)
        predictions.extend(fold_predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = [asdict(row) for row in metrics]
    if metrics:
        mae = float(np.nanmean([row.mae for row in metrics]))
        rmse = float(np.nanmean([row.rmse for row in metrics]))
        r2 = float(np.nanmean([row.r2 for row in metrics]))
        ae = float(np.nanmean([row.AE for row in metrics]))
        re = float(np.nanmean([row.RE for row in metrics]))
        persistence_mae = float(np.nanmean([row.persistence_mae for row in metrics]))
        linear_mae = float(np.nanmean([row.linear_last5_mae for row in metrics]))
    else:
        mae = rmse = r2 = ae = re = persistence_mae = linear_mae = float("nan")

    payload: Dict[str, object] = {
        "protocol": "patchformer",
        "task": f"past_{config.seq_len}_cycles_to_next_capacity",
        "preprocessing": {
            "macro_indicator_imputation": "per-cell causal forward fill; leading missing values become 0.0",
            "normalization": "capacity min-max and feature standardizer are fit on training batteries only",
            "train_test_split": "leave-one-battery-out; held-out battery excluded from model fitting",
            "window_policy": "history cycles [target_cycle-seq_len, target_cycle-1] only",
        },
        "config": asdict(config),
        "folds": metrics_payload,
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "AE": ae,
            "RE": re,
            "persistence_mae": persistence_mae,
            "linear_last5_mae": linear_mae,
            "num_fold_startpoint_runs": len(metrics),
            "num_prediction_windows": len(predictions),
        },
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    pred_path = output_dir / "predictions.csv"
    if predictions:
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
            writer.writeheader()
            writer.writerows(predictions)
    return payload


def format_results(payload: Dict[str, object]) -> str:
    folds = payload["folds"]
    metrics = payload["metrics"]
    lines = []
    header = (
        "battery".ljust(12)
        + " start".rjust(8)
        + " windows".rjust(10)
        + " scale".rjust(8)
        + " mae".rjust(12)
        + " rmse".rjust(12)
        + " r2".rjust(10)
        + " AE".rjust(8)
        + " RE".rjust(10)
        + " persist".rjust(12)
        + " linear5".rjust(12)
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in folds:  # type: ignore[assignment]
        lines.append(
            str(row["test_battery"]).ljust(12)
            + f"{row['start_point']:8d}"
            + f"{row['num_windows']:10d}"
            + f"{row['residual_scale']:8.2f}"
            + f"{row['mae']:12.6f}"
            + f"{row['rmse']:12.6f}"
            + f"{row['r2']:10.4f}"
            + f"{row['AE']:8d}"
            + f"{row['RE']:10.4f}"
            + f"{row['persistence_mae']:12.6f}"
            + f"{row['linear_last5_mae']:12.6f}"
        )
    lines.append("")
    lines.append(
        "Overall: "
        f"MAE={metrics['mae']:.6f} Ah, "
        f"RMSE={metrics['rmse']:.6f} Ah, "
        f"R2={metrics['r2']:.4f}, "
        f"AE={metrics['AE']:.2f}, "
        f"RE={metrics['RE']:.4f}, "
        f"Persistence MAE={metrics['persistence_mae']:.6f} Ah, "
        f"LinearLast5 MAE={metrics['linear_last5_mae']:.6f} Ah"
    )
    return "\n".join(lines)
