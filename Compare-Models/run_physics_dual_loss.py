"""Run the MSTEA-Net dual-physics-loss reproduction on the aligned NASA task."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACKAGE = HERE / "PhysicsDualLoss"
for path in (ROOT, PACKAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from comparison_protocol import (  # noqa: E402
    DEFAULT_SEEDS, PROTOCOLS, evaluate_predictions, load_summary,
    protocol_manifest, seed_everything, write_csv, write_json,
)
from features import (  # noqa: E402
    RAW_FEATURES, FeatureSelection, decompose_features, detect_period, extract_nasa_health_features,
    extract_summary_health_features,
    select_wrapper_features,
)
from model import MSTEANet, TripleCompositeLoss  # noqa: E402


@dataclass(frozen=True)
class PaperConfig:
    # The paper reports 0.01 for CALCE/TJU.  Dataset adapters may replace it
    # only through training-cell validation; the held-out test cell is never
    # consulted.
    learning_rate: float = 3e-4
    window_length: int = 16
    hidden_dimension: int = 32
    attention_heads: int = 4
    epochs: int = 200
    lambda_arr: float = 1e-4
    lambda_deriv: float = 1e-4
    activation_energy_ev: float = 0.65
    degradation_exponent: float = 1.5
    temperature_kelvin: float = 298.15
    batch_size: int = 16
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-6
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = None
    feature_selection_mode: str = "training_wrapper"
    feature_selection_seed: int = 7
    capacity_representation: str = "raw_ah"
    refit_full_training: bool = True


PAPER_TRAINING_FEATURES = {
    # Published Table 2 majority vote over #36/#37/#38; held-out #35 is
    # excluded. CCDTTrend/CCCTTrend occur in 3/3 and CVCTTrend in 2/3.
    "calce": ("CCDTTrend", "CCCTTrend", "CVCTTrend"),
    # Published Table 2 union over #2/#3; held-out #1 is excluded.
    "tju": ("CCDTTrend", "CCCTTrend", "CVCTResidual"),
}


VALIDATION_SELECTED_LR = {"nasa": 1e-3, "calce": 3e-4, "tju": 3e-4}


@dataclass
class WindowBatch:
    x: torch.Tensor
    y: torch.Tensor
    cycle: torch.Tensor
    initial_capacity: torch.Tensor
    group: torch.Tensor

    def to(self, device: torch.device) -> "WindowBatch":
        return WindowBatch(*(value.to(device) for value in (
            self.x, self.y, self.cycle, self.initial_capacity, self.group
        )))


def _split_index(length: int, seq_len: int) -> int:
    split = int(math.floor(length * 0.8))
    if split <= seq_len:
        raise ValueError(f"sequence length {length} is too short for window {seq_len}")
    return split


def make_batch(names: list[str], starts: dict[str, int], stops: dict[str, int],
               values: dict[str, np.ndarray], frames, seq_len: int,
               feature_mean: np.ndarray, feature_std: np.ndarray) -> WindowBatch:
    windows, targets, cycles, c0s, groups = [], [], [], [], []
    for group_index, name in enumerate(names):
        capacity = frames[name]["Capacity"].to_numpy(dtype=np.float32)
        cycle = frames[name]["Cycle"].to_numpy(dtype=np.float32)
        normalized = (values[name] - feature_mean) / feature_std
        for target in range(max(starts[name], seq_len), stops[name]):
            windows.append(normalized[target - seq_len:target])
            # Paper Eqs. (31) and (36) optimize absolute capacity C(N), while
            # the physics term normalizes degradation using the true C(0).
            targets.append(capacity[target])
            cycles.append(cycle[target])
            c0s.append(capacity[0])
            groups.append(group_index)
    if not windows:
        raise ValueError("no windows were constructed")
    return WindowBatch(
        torch.as_tensor(np.asarray(windows), dtype=torch.float32),
        torch.as_tensor(targets, dtype=torch.float32),
        torch.as_tensor(cycles, dtype=torch.float32),
        torch.as_tensor(c0s, dtype=torch.float32),
        torch.as_tensor(groups, dtype=torch.long),
    )


@torch.no_grad()
def evaluate_loss(model, objective, batch) -> dict[str, float]:
    model.eval()
    components = objective.components(
        model(batch.x), batch.y, batch.cycle, batch.initial_capacity, batch.group
    )
    return {name: float(value.detach().cpu()) for name, value in components.items()}


def train(model, objective, training: WindowBatch, validation: WindowBatch,
          epochs: int, learning_rate: float, batch_size: int, seed: int, tag: str,
          patience: int, min_delta: float, weight_decay: float = 0.0,
          gradient_clip_norm: float | None = None):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state, best_value, best_epoch = None, float("inf"), 0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    # Preserve Eq. (35): every mini-batch is a contiguous run from one cell.
    blocks: list[torch.Tensor] = []
    for group_value in torch.unique(training.group, sorted=True):
        indices = torch.nonzero(training.group == group_value, as_tuple=False).flatten()
        blocks.extend(indices[start:start + batch_size]
                      for start in range(0, len(indices), batch_size))
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {name: 0.0 for name in ("total", "data", "arrhenius", "derivative")}
        order = torch.randperm(
            len(blocks), generator=torch.Generator().manual_seed(seed + epoch)
        ).tolist()
        for block_index in order:
            indices = blocks[block_index]
            optimizer.zero_grad(set_to_none=True)
            components = objective.components(
                model(training.x[indices]), training.y[indices], training.cycle[indices],
                training.initial_capacity[indices], training.group[indices],
            )
            components["total"].backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            for name, value in components.items():
                totals[name] += float(value.detach().cpu()) * len(indices)
        training_components = {name: value / len(training.x) for name, value in totals.items()}
        validation_components = evaluate_loss(model, objective, validation)
        row = {
            "epoch": epoch,
            **{f"train_{name}": value for name, value in training_components.items()},
            **{f"validation_{name}": value for name, value in validation_components.items()},
        }
        history.append(row)
        print(
            f"{tag} epoch={epoch:03d}/{epochs} train={row['train_total']:.8f} "
            f"data={row['train_data']:.8f} arr={row['train_arrhenius']:.8f} "
            f"deriv={row['train_derivative']:.8f} val={row['validation_total']:.8f}",
            flush=True,
        )
        if row["validation_total"] < best_value - min_delta:
            best_value = float(row["validation_total"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"{tag} early_stop epoch={epoch:03d} best_epoch={best_epoch:03d} "
                    f"best_val={best_value:.8f} patience={patience}",
                    flush=True,
                )
                break
    if best_state is None:
        raise RuntimeError("training produced no finite checkpoint")
    model.load_state_dict(best_state)
    return history, best_epoch, best_value


def train_fixed_epochs(model, objective, training: WindowBatch, epochs: int,
                       learning_rate: float, batch_size: int, seed: int, tag: str,
                       weight_decay: float = 0.0,
                       gradient_clip_norm: float | None = None):
    """Refit on every non-test-cell window after leakage-free epoch selection."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    blocks: list[torch.Tensor] = []
    for group_value in torch.unique(training.group, sorted=True):
        indices = torch.nonzero(training.group == group_value, as_tuple=False).flatten()
        blocks.extend(indices[start:start + batch_size]
                      for start in range(0, len(indices), batch_size))
    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {name: 0.0 for name in ("total", "data", "arrhenius", "derivative")}
        order = torch.randperm(
            len(blocks), generator=torch.Generator().manual_seed(seed + epoch)
        ).tolist()
        for block_index in order:
            indices = blocks[block_index]
            optimizer.zero_grad(set_to_none=True)
            components = objective.components(
                model(training.x[indices]), training.y[indices], training.cycle[indices],
                training.initial_capacity[indices], training.group[indices],
            )
            components["total"].backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            for name, value in components.items():
                totals[name] += float(value.detach().cpu()) * len(indices)
        row = {"epoch": epoch, **{
            f"train_{name}": value / len(training.x) for name, value in totals.items()
        }}
        history.append(row)
        print(
            f"{tag} epoch={epoch:03d}/{epochs} train={row['train_total']:.8f} "
            f"data={row['train_data']:.8f} arr={row['train_arrhenius']:.8f} "
            f"deriv={row['train_derivative']:.8f}", flush=True,
        )
    return history


def run_fold(dataset: str, seed: int, output_root: Path, device: torch.device,
             max_epochs: int | None = None) -> None:
    protocol = PROTOCOLS[dataset]
    test_battery = protocol.batteries[0]
    output = output_root / "physics-dual-loss" / dataset / test_battery / f"seed_{seed}"
    result_path = output / "results.json"
    epochs = int(max_epochs or 200)
    # The paper's primary feature-selection method is the Wrapper pipeline.
    # CALCE validation confirms that fitting it on training segments generalizes
    # better than freezing the union of Table 2's reported per-cell subsets.
    feature_selection_mode = (
        "paper_table2_training_cell_union" if dataset == "tju"
        else "training_wrapper"
    )
    learning_rate = VALIDATION_SELECTED_LR[dataset]
    if result_path.exists():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            prior_config = existing.get("paper_config", {})
            if (existing.get("status") == "complete"
                    and prior_config.get("epochs") == epochs
                    and prior_config.get("early_stopping_patience") == 20
                    and prior_config.get("early_stopping_min_delta") == 1e-6
                    and prior_config.get("learning_rate") == learning_rate
                    and prior_config.get("feature_selection_mode") == feature_selection_mode
                    and prior_config.get("feature_selection_seed") == 7
                    and prior_config.get("weight_decay", 0.0) == 0.0
                    and prior_config.get("gradient_clip_norm") is None
                    and prior_config.get("capacity_representation") == "raw_ah"
                    and prior_config.get("refit_full_training") is True):
                print(f"skip complete: {output}", flush=True)
                return
        except (OSError, json.JSONDecodeError):
            pass

    seed_everything(seed)
    summaries = load_summary(protocol)
    if dataset == "nasa":
        health = extract_nasa_health_features(ROOT / "data/version3/NASA data", summaries)
        raw_features = RAW_FEATURES
        raw_feature_source = "official NASA MAT records in data/version3/NASA data"
    else:
        health, raw_features = extract_summary_health_features(dataset, summaries)
        raw_feature_source = f"read-only aligned {dataset.upper()} cycle summaries under data/version3"
    train_names = [name for name in protocol.batteries if name != test_battery]
    split_indices = {name: _split_index(len(health[name]), protocol.seq_len) for name in train_names}
    # Period detection uses training-cell capacity only; the held-out cell
    # trajectory never influences feature construction or model selection.
    periods = [detect_period(health[name]["Capacity"].to_numpy()[:split_indices[name]])
               for name in train_names]
    period = int(np.median(periods))
    decomposed, feature_names = decompose_features(
        health, period, relative_to_initial=False, raw_features=raw_features
    )
    if feature_selection_mode == "paper_table2_training_cell_union":
        names = PAPER_TRAINING_FEATURES[dataset]
        indices = tuple(feature_names.index(name) for name in names)
        selection = FeatureSelection(indices, names, (), (), names)
    else:
        selection = select_wrapper_features(
            decomposed, health, train_names, feature_names, split_indices, 7
        )
    # Section 4.1 uses sliding windows from the target degradation sequence.
    # Append observed raw capacity as an input channel; it is never copied or
    # added at the output.  The prediction head still directly estimates C(N).
    selected = {
        name: np.column_stack((
            values[:, selection.indices],
            health[name]["Capacity"].to_numpy(dtype=np.float32),
        )).astype(np.float32)
        for name, values in decomposed.items()
    }
    fit_values = np.concatenate([selected[name][:split_indices[name]] for name in train_names])
    feature_mean = fit_values.mean(axis=0).astype(np.float32)
    feature_std = fit_values.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-8] = 1.0

    zeros = {name: 0 for name in train_names}
    training = make_batch(
        train_names, zeros, split_indices, selected, health, protocol.seq_len,
        feature_mean, feature_std,
    ).to(device)
    validation = make_batch(
        train_names, split_indices, {name: len(health[name]) for name in train_names},
        selected, health, protocol.seq_len, feature_mean, feature_std,
    ).to(device)
    test = make_batch(
        [test_battery], {test_battery: 0}, {test_battery: len(health[test_battery])},
        selected, health, protocol.seq_len, feature_mean, feature_std,
    ).to(device)

    config = PaperConfig(
        learning_rate=learning_rate,
        window_length=protocol.seq_len,
        epochs=epochs,
        temperature_kelvin=274.15 if dataset == "calce" else 298.15,
        feature_selection_mode=feature_selection_mode,
    )
    model = MSTEANet(selected[test_battery].shape[1], config.hidden_dimension,
                     config.attention_heads).to(device)
    objective = TripleCompositeLoss(
        config.temperature_kelvin, config.activation_energy_ev,
        config.degradation_exponent, config.lambda_arr, config.lambda_deriv,
    ).to(device)
    started = time.time()
    history, best_epoch, best_validation = train(
        model, objective, training, validation, epochs, config.learning_rate,
        config.batch_size, seed,
        f"[physics-dual-loss/{dataset}/{test_battery}/seed={seed}]",
        config.early_stopping_patience, config.early_stopping_min_delta,
        config.weight_decay, config.gradient_clip_norm,
    )

    # The paper trains each LOO fold with all N-1 non-test cells. Validation
    # above selects the epoch without touching the held-out cell; the final
    # model is then refitted from scratch using every allowed training window.
    full_stops = {name: len(health[name]) for name in train_names}
    full_periods = [detect_period(health[name]["Capacity"].to_numpy())
                    for name in train_names]
    final_period = int(np.median(full_periods))
    final_decomposed, final_feature_names = decompose_features(
        health, final_period, relative_to_initial=False, raw_features=raw_features
    )
    if feature_selection_mode == "paper_table2_training_cell_union":
        final_names = PAPER_TRAINING_FEATURES[dataset]
        final_indices = tuple(final_feature_names.index(name) for name in final_names)
        final_selection = FeatureSelection(final_indices, final_names, (), (), final_names)
    else:
        final_selection = select_wrapper_features(
            final_decomposed, health, train_names, final_feature_names, full_stops, 7
        )
    final_selected = {
        name: np.column_stack((
            values[:, final_selection.indices],
            health[name]["Capacity"].to_numpy(dtype=np.float32),
        )).astype(np.float32)
        for name, values in final_decomposed.items()
    }
    final_fit_values = np.concatenate([final_selected[name] for name in train_names])
    final_feature_mean = final_fit_values.mean(axis=0).astype(np.float32)
    final_feature_std = final_fit_values.std(axis=0).astype(np.float32)
    final_feature_std[final_feature_std < 1e-8] = 1.0
    final_training = make_batch(
        train_names, zeros, full_stops, final_selected, health, protocol.seq_len,
        final_feature_mean, final_feature_std,
    ).to(device)
    test = make_batch(
        [test_battery], {test_battery: 0}, {test_battery: len(health[test_battery])},
        final_selected, health, protocol.seq_len, final_feature_mean, final_feature_std,
    ).to(device)
    seed_everything(seed)
    model = MSTEANet(final_selected[test_battery].shape[1], config.hidden_dimension,
                     config.attention_heads).to(device)
    objective = TripleCompositeLoss(
        config.temperature_kelvin, config.activation_energy_ev,
        config.degradation_exponent, config.lambda_arr, config.lambda_deriv,
    ).to(device)
    refit_history = train_fixed_epochs(
        model, objective, final_training, best_epoch, config.learning_rate,
        config.batch_size, seed,
        f"[physics-dual-loss-refit/{dataset}/{test_battery}/seed={seed}]",
        config.weight_decay, config.gradient_clip_norm,
    )
    model.eval()
    with torch.no_grad():
        prediction = model(test.x).detach().cpu().numpy()
    y_true = test.y.detach().cpu().numpy()
    cycles = test.cycle.detach().cpu().numpy().astype(np.int64)
    metrics, prediction_rows = evaluate_predictions(cycles, y_true, prediction, protocol)
    for row in prediction_rows:
        row.update({"model": "physics-dual-loss", "dataset": dataset,
                    "battery": test_battery, "seed": seed})
    result = {
        "status": "complete", "model": "physics-dual-loss", "paper_model": "MSTEA-Net",
        "paper_doi": "10.1016/j.energy.2026.140288", "dataset": dataset,
        "test_battery": test_battery, "train_batteries": train_names, "seed": seed,
        "task": f"{protocol.seq_len}-cycle rolling one-step capacity prediction",
        "start_points": list(protocol.start_points), "metrics": metrics,
        "best_epoch": best_epoch, "best_validation_total": best_validation,
        "epochs_trained": len(refit_history), "stopped_early": len(history) < epochs,
        "inner_epochs_trained": len(history),
        "elapsed_seconds": time.time() - started,
        "paper_config": asdict(config),
        "arrhenius_rate": objective.arrhenius_rate,
        "adaptive_period": final_period, "training_cell_periods": full_periods,
        "inner_adaptive_period": period, "inner_training_cell_periods": periods,
        "feature_selection": asdict(final_selection),
        "feature_selection_seed": config.feature_selection_seed,
        "feature_selection_evidence": (
            "published Table 2 union over unified-protocol training cells only"
            if feature_selection_mode == "paper_table2_training_cell_union"
            else "Wrapper refitted on all non-test-cell sequences after inner epoch selection"
        ),
        "model_input_features": [*final_selection.names, "ObservedCapacityAh"],
        "normalization": {"type": "all non-test-cell feature standardization after inner epoch selection; raw Ah target",
                          "mean": final_feature_mean.tolist(), "std": final_feature_std.tolist()},
        "processed_summary": protocol.summary,
        "raw_feature_source": raw_feature_source,
        "adaptations": [
            f"paper window 64 changed to unified {dataset.upper()} window {protocol.seq_len}" if protocol.seq_len != 64 else "paper window 64 retained",
            f"temperature set to {config.temperature_kelvin} K for {dataset.upper()}",
            "capacity target retained in raw Ah as in paper Eqs. (31) and (36)",
            "health features retain original units before STL and are standardized using training segments",
            f"feature selection and normalization exclude held-out {test_battery}",
            f"learning rate {learning_rate:g} selected using training-cell validation only",
            "training duration selected by inner 80/20 validation, then refitted from scratch on all non-test-cell windows",
            "STL endpoint calculation is causal to satisfy rolling online prediction",
            f"one {test_battery} holdout follows the unified benchmark instead of full LOO-CV",
            "observed raw-capacity history is an input channel under paper Section 4.1; the head outputs capacity directly without persistence addition",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "result": result}, output / "checkpoint.pt")
    write_csv(output / "training_history.csv", refit_history)
    write_csv(output / "inner_validation_history.csv", history)
    write_csv(output / "predictions.csv", prediction_rows)
    write_json(result_path, result)
    print(f"complete: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(PROTOCOLS), default="nasa")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs/comparison_models_10seeds")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Optional override; default maximum is 200 for every dataset.")
    args = parser.parse_args()
    device = torch.device(args.device)
    write_json(args.output_root / "protocol.json", protocol_manifest())
    for seed in args.seeds:
        run_fold(args.dataset, seed, args.output_root.resolve(), device, args.max_epochs)


if __name__ == "__main__":
    main()
