"""Validation-only learning-rate audit for the MSTEA-Net reproduction.

The held-out benchmark battery is never evaluated by this script.  Every
candidate retains the paper architecture, window, feature-selection pipeline,
and triple-composite loss; only optimization hyperparameters are compared on
the trailing validation segments of the training batteries.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "Compare-Models"
PACKAGE = COMPARE / "PhysicsDualLoss"
for path in (ROOT, COMPARE, PACKAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_physics_dual_loss as base  # noqa: E402


PAPER_TRAINING_CONSENSUS = {
    # Constructed only from the published Table 2 rows belonging to the
    # training cells of the unified outer holdout.  The held-out row is not
    # consulted, so this does not use test-cell labels for model selection.
    # Majority vote over the published training-cell rows #36/#37/#38.
    # The held-out #35 row is excluded: CCDTTrend/CCCTTrend occur in 3/3
    # training rows and CVCTTrend in 2/3.
    "calce": ("CCDTTrend", "CCCTTrend", "CVCTTrend"),
    "tju": ("CCDTTrend", "CCCTTrend", "CVCTResidual"),
}


def prepare(dataset: str, feature_selection_seed: int, device: torch.device,
            feature_mode: str):
    protocol = base.PROTOCOLS[dataset]
    held_out = protocol.batteries[0]
    summaries = base.load_summary(protocol)
    if dataset == "nasa":
        health = base.extract_nasa_health_features(
            ROOT / "data/version3/NASA data", summaries
        )
        raw_features = base.RAW_FEATURES
    else:
        health, raw_features = base.extract_summary_health_features(dataset, summaries)
    train_names = [name for name in protocol.batteries if name != held_out]
    split_indices = {
        name: base._split_index(len(health[name]), protocol.seq_len)
        for name in train_names
    }
    periods = [
        base.detect_period(
            health[name]["Capacity"].to_numpy()[: split_indices[name]]
        )
        for name in train_names
    ]
    period = int(np.median(periods))
    decomposed, feature_names = base.decompose_features(
        health, period, relative_to_initial=False, raw_features=raw_features
    )
    if feature_mode == "wrapper":
        selection = base.select_wrapper_features(
            decomposed, health, train_names, feature_names, split_indices,
            feature_selection_seed,
        )
        selection_indices = selection.indices
        selection_names = selection.names
    elif feature_mode == "paper-consensus":
        selection_names = PAPER_TRAINING_CONSENSUS[dataset]
        selection_indices = tuple(feature_names.index(name) for name in selection_names)
    else:
        raise ValueError(feature_mode)
    selected = {
        name: np.column_stack(
            (
                values[:, selection_indices],
                health[name]["Capacity"].to_numpy(dtype=np.float32),
            )
        ).astype(np.float32)
        for name, values in decomposed.items()
    }
    fit_values = np.concatenate(
        [selected[name][: split_indices[name]] for name in train_names]
    )
    feature_mean = fit_values.mean(axis=0).astype(np.float32)
    feature_std = fit_values.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-8] = 1.0
    zeros = {name: 0 for name in train_names}
    training = base.make_batch(
        train_names,
        zeros,
        split_indices,
        selected,
        health,
        protocol.seq_len,
        feature_mean,
        feature_std,
    ).to(device)
    validation = base.make_batch(
        train_names,
        split_indices,
        {name: len(health[name]) for name in train_names},
        selected,
        health,
        protocol.seq_len,
        feature_mean,
        feature_std,
    ).to(device)
    return training, validation, len(selection_indices) + 1, selection_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=("nasa", "calce", "tju"), default=["calce", "tju"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-2, 3e-3, 1e-3, 3e-4])
    parser.add_argument("--feature-modes", nargs="+", choices=("wrapper", "paper-consensus"), default=["wrapper", "paper-consensus"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16])
    parser.add_argument("--feature-selection-seed", type=int, default=7)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/physics_dual_loss_validation_hpo")
    args = parser.parse_args()
    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        temperature = 274.15 if dataset == "calce" else 298.15
        for seed in args.seeds:
            for feature_mode in args.feature_modes:
                base.seed_everything(seed)
                training, validation, input_dimension, feature_names = prepare(
                    dataset, args.feature_selection_seed, device, feature_mode
                )
                for batch_size in args.batch_sizes:
                    for learning_rate in args.learning_rates:
                        base.seed_everything(seed)
                        model = base.MSTEANet(input_dimension, 32, 4).to(device)
                        objective = base.TripleCompositeLoss(
                            temperature, 0.65, 1.5, 1e-4, 1e-4
                        ).to(device)
                        log_path = args.output_root / f"{dataset}_{feature_mode}_seed{seed}_lr{learning_rate:g}_bs{batch_size}.log"
                        with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                            history, best_epoch, best_value = base.train(
                                model,
                                objective,
                                training,
                                validation,
                                args.max_epochs,
                                learning_rate,
                                batch_size,
                                seed,
                                f"[selection/{dataset}/{feature_mode}/seed={seed}/lr={learning_rate:g}/bs={batch_size}]",
                                20,
                                1e-6,
                                args.weight_decay,
                                args.gradient_clip_norm,
                            )
                        row = {
                            "dataset": dataset,
                            "seed": seed,
                            "feature_mode": feature_mode,
                            "feature_selection_seed": args.feature_selection_seed,
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                            "weight_decay": args.weight_decay,
                            "gradient_clip_norm": args.gradient_clip_norm,
                            "best_epoch": best_epoch,
                            "best_validation_total": best_value,
                            "epochs_trained": len(history),
                            "input_dimension": input_dimension,
                            "selected_features": list(feature_names),
                            "test_evaluated": False,
                        }
                        rows.append(row)
                        print(json.dumps(row, ensure_ascii=False), flush=True)
                        (args.output_root / "trials.json").write_text(
                            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
    summary: dict[str, dict[str, float]] = {}
    for dataset in args.datasets:
        candidates = {}
        for feature_mode in args.feature_modes:
            for batch_size in args.batch_sizes:
                for learning_rate in args.learning_rates:
                    values = [
                        float(row["best_validation_total"])
                        for row in rows
                        if row["dataset"] == dataset
                        and row["feature_mode"] == feature_mode
                        and row["learning_rate"] == learning_rate
                        and row["batch_size"] == batch_size
                    ]
                    candidates[f"{feature_mode}|{learning_rate:g}|{batch_size}"] = float(np.mean(values))
        best_key = min(candidates, key=candidates.get)
        best_mode, best_lr, best_batch = best_key.split("|")
        summary[dataset] = {
            "selected_feature_mode": best_mode,
            "selected_learning_rate": float(best_lr),
            "selected_batch_size": int(best_batch),
            "mean_validation_total": candidates[best_key],
            "candidates": candidates,
        }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
