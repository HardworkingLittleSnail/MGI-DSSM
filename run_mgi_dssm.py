from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mgi_dssm.data import build_feature_frame, load_cycle_summary
from mgi_dssm.physics_train import PhysicsTrainConfig, format_physics_results, run_physics_mgi
from mgi_dssm.raw_calce import audit_raw_files
from mgi_dssm.residual_boosting import (
    ResidualBoostingConfig,
    format_residual_results,
    run_residual_boosting,
)
from mgi_dssm.train import TrainConfig, format_results, run_leave_one_battery_out


PATH_CONFIG_KEYS = {"data_dir", "output_dir"}


def _option_dests(parser: argparse.ArgumentParser) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            mapping[option] = action.dest
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                mapping.update(_option_dests(subparser))
    return mapping


def _supplied_cli_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_to_dest = _option_dests(parser)
    supplied: set[str] = set()
    for token in argv:
        option = token.split("=", 1)[0]
        if option in option_to_dest:
            supplied.add(option_to_dest[option])
    return supplied


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    config = payload.get("train", payload)
    if not isinstance(config, dict):
        raise ValueError(f"Config field 'train' must be an object: {path}")
    return {str(key).replace("-", "_"): value for key, value in config.items()}


def _apply_config(args: argparse.Namespace, parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    config_path = getattr(args, "config", None)
    if config_path is None:
        return args

    config = _load_config(Path(config_path))
    supplied = _supplied_cli_dests(parser, argv)
    valid_keys = set(vars(args))
    unknown = sorted(set(config).difference(valid_keys))
    if unknown:
        raise ValueError(f"Unknown config key(s) in {config_path}: {unknown}")

    for key, value in config.items():
        if key in supplied or key == "config":
            continue
        if key in PATH_CONFIG_KEYS and value is not None:
            value = Path(value)
        setattr(args, key, value)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the lightweight MGI-DSSM battery model.")
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help="Train and evaluate leave-one-battery-out folds.")
    train.add_argument("--config", type=Path, default=None, help="JSON config file for train arguments.")
    train.add_argument("--data-dir", type=Path, default=Path("data"))
    train.add_argument("--dataset", choices=["calce", "nasa", "tju"], default="calce")
    train.add_argument("--output-dir", type=Path, default=Path("outputs") / "mgi_dssm_lite")
    train.add_argument("--seq-len", type=int, default=64)
    train.add_argument("--head", choices=["mgi-lite", "mgi-physics", "residual-hgb"], default="mgi-lite")
    train.add_argument("--max-seq-len", type=int, default=1000)
    train.add_argument("--rated-capacity", type=float, default=1.1)
    train.add_argument("--start-points", type=int, nargs="+", default=[65])
    train.add_argument("--test-names", nargs="*", default=None, help="Optional held-out batteries, e.g. CS2_35 CS2_37.")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--recon-weight", type=float, default=0.05)
    train.add_argument("--residual-anchor-weight", type=float, default=0.02)
    train.add_argument("--huber-beta", type=float, default=0.005)
    train.add_argument("--hgb-max-iter", type=int, default=300)
    train.add_argument("--hgb-learning-rate", type=float, default=0.02)
    train.add_argument("--hgb-max-leaf-nodes", type=int, default=12)
    train.add_argument("--hgb-l2", type=float, default=0.1)
    train.add_argument(
        "--residual-feature-mode",
        choices=["capacity", "calce-summary"],
        default="calce-summary",
        help="calce-summary uses Capacity/Resistance/CCCT/CVCT; capacity is a capacity-only ablation.",
    )
    train.add_argument("--physics-num-layers", type=int, default=1)
    train.add_argument("--physics-weight-decay", type=float, default=1e-4)
    train.add_argument("--physics-adam-beta1", type=float, default=0.9)
    train.add_argument("--physics-adam-beta2", type=float, default=0.999)
    train.add_argument("--physics-adam-eps", type=float, default=1e-8)
    train.add_argument(
        "--physics-lr-scheduler", choices=["none", "cosine", "plateau"], default="none"
    )
    train.add_argument("--physics-scheduler-min-lr-ratio", type=float, default=0.05)
    train.add_argument("--physics-scheduler-patience", type=int, default=6)
    train.add_argument("--physics-capacity-huber-beta", type=float, default=0.01)
    train.add_argument("--physics-grad-clip-norm", type=float, default=1.0)
    train.add_argument("--physics-validation-fraction", type=float, default=0.15)
    train.add_argument("--physics-early-stopping-patience", type=int, default=12)
    train.add_argument("--physics-state-loss-weight", type=float, default=0.4)
    train.add_argument(
        "--physics-state-supervision",
        choices=["coordinate", "curve"],
        default="coordinate"
    )
    train.add_argument("--physics-curve-loss-weight", type=float, default=0.02)
    train.add_argument("--physics-weak-state-loss-weight", type=float, default=0.02)
    train.add_argument("--physics-voltage-error-scale", type=float, default=0.05)
    train.add_argument("--physics-capacity-loss-weight", type=float, default=1.0)
    train.add_argument("--physics-regeneration-loss-weight", type=float, default=0.0)
    train.add_argument("--physics-direction-loss-weight", type=float, default=0.05)
    train.add_argument("--physics-late-life-weight", type=float, default=0.5)
    train.add_argument("--physics-cutoff-voltage", type=float, default=2.7)
    train.add_argument("--physics-discharge-current", type=float, default=1.1)
    train.add_argument("--physics-tau-p-seconds", type=float, default=120.0)
    train.add_argument("--physics-q-grid-max-ah", type=float, default=1.5)
    train.add_argument("--physics-q-grid-points", type=int, default=400)
    train.add_argument("--physics-max-self-reconstruction-mae", type=float, default=0.008)
    train.add_argument("--physics-cache-name", type=str, default="physics_curve_cache_v1.npz")
    train.add_argument("--physics-summary-filename", type=str, default=None)
    train.add_argument(
        "--physics-ocp-profile",
        choices=["lco_graphite", "nmc_graphite_siox"],
        default="lco_graphite",
    )
    train.add_argument("--physics-outlier-sigma-window", type=int, default=0)
    train.add_argument("--physics-outlier-preserve-endpoints", action="store_true")
    train.add_argument(
        "--physics-preprocessing-protocol",
        choices=["legacy", "batter_moe"],
        default="legacy",
        help=(
            "batter_moe uses offline isolated-3-sigma cleaning, C/C0 capacity "
            "targets, and train-only Min-Max scaling for state features."
        ),
    )
    train.add_argument(
        "--physics-state-scaling",
        choices=["protocol", "minmax", "zscore"],
        default="protocol",
        help="Override latent-state scaling without changing raw-data preprocessing.",
    )
    train.add_argument(
        "--physics-capacity-target-scaling",
        choices=["protocol", "soh", "absolute"],
        default="protocol",
        help="Override capacity-loss units without changing raw-data preprocessing.",
    )
    train.add_argument("--physics-thermo-step-scale", type=float, default=0.02)
    train.add_argument("--physics-kinetic-step-scale", type=float, default=0.03)
    train.add_argument("--physics-trend-short-window", type=int, default=8)
    train.add_argument("--physics-trend-long-window", type=int, default=32)
    train.add_argument(
        "--physics-evaluation-protocol",
        choices=["patchformer", "mstea"],
        default="patchformer",
    )
    train.add_argument("--physics-threshold-bias-calibration-soh", type=float, default=0.0)
    train.add_argument("--physics-threshold-bias-band-soh", type=float, default=0.015)
    train.add_argument(
        "--physics-threshold-bias-mode",
        choices=["symmetric"],
        default="symmetric",
    )
    train.add_argument(
        "--physics-eol-event-phase-alignment",
        choices=["none", "global", "robust"],
        default="none",
    )
    train.add_argument("--physics-eol-event-phase-clip-cycles", type=int, default=2)

    audit = sub.add_parser("audit-raw", help="Audit a few raw CALCE XLSX files with the stdlib parser.")
    audit.add_argument("--data-dir", type=Path, default=Path("data"))
    audit.add_argument("--max-files", type=int, default=4)

    parser.set_defaults(command="train")
    args = parser.parse_args()
    if args.command == "train":
        args = _apply_config(args, parser, sys.argv[1:])
    return args


def main() -> None:
    args = parse_args()
    if args.command == "audit-raw":
        payload = audit_raw_files(args.data_dir, max_files=args.max_files)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if args.head == "mgi-physics":
        if args.dataset not in {"calce", "nasa", "tju"}:
            raise ValueError("mgi-physics requires raw CALCE XLSX, NASA MAT, or TJU CSV curves.")
        config = PhysicsTrainConfig(
            dataset=args.dataset,
            seq_len=args.seq_len,
            rated_capacity=args.rated_capacity,
            start_points=tuple(args.start_points),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.physics_weight_decay,
            adam_beta1=args.physics_adam_beta1,
            adam_beta2=args.physics_adam_beta2,
            adam_eps=args.physics_adam_eps,
            lr_scheduler=args.physics_lr_scheduler,
            scheduler_min_lr_ratio=args.physics_scheduler_min_lr_ratio,
            scheduler_patience=args.physics_scheduler_patience,
            capacity_huber_beta=args.physics_capacity_huber_beta,
            grad_clip_norm=args.physics_grad_clip_norm,
            hidden_dim=args.hidden_dim,
            num_layers=args.physics_num_layers,
            dropout=args.dropout,
            seed=args.seed,
            validation_fraction=args.physics_validation_fraction,
            early_stopping_patience=args.physics_early_stopping_patience,
            state_loss_weight=args.physics_state_loss_weight,
            state_supervision=args.physics_state_supervision,
            curve_loss_weight=args.physics_curve_loss_weight,
            weak_state_loss_weight=args.physics_weak_state_loss_weight,
            voltage_error_scale=args.physics_voltage_error_scale,
            capacity_loss_weight=args.physics_capacity_loss_weight,
            regeneration_loss_weight=args.physics_regeneration_loss_weight,
            direction_loss_weight=args.physics_direction_loss_weight,
            late_life_weight=args.physics_late_life_weight,
            cutoff_voltage_v=args.physics_cutoff_voltage,
            discharge_current_a=args.physics_discharge_current,
            tau_p_seconds=args.physics_tau_p_seconds,
            q_grid_max_ah=args.physics_q_grid_max_ah,
            q_grid_points=args.physics_q_grid_points,
            max_self_reconstruction_mae=args.physics_max_self_reconstruction_mae,
            cache_name=args.physics_cache_name,
            summary_filename=args.physics_summary_filename,
            ocp_profile=args.physics_ocp_profile,
            outlier_sigma_window=args.physics_outlier_sigma_window,
            outlier_preserve_endpoints=args.physics_outlier_preserve_endpoints,
            preprocessing_protocol=args.physics_preprocessing_protocol,
            state_scaling=args.physics_state_scaling,
            capacity_target_scaling=args.physics_capacity_target_scaling,
            thermo_step_scale=args.physics_thermo_step_scale,
            kinetic_step_scale=args.physics_kinetic_step_scale,
            trend_short_window=args.physics_trend_short_window,
            trend_long_window=args.physics_trend_long_window,
            evaluation_protocol=args.physics_evaluation_protocol,
            threshold_bias_calibration_soh=args.physics_threshold_bias_calibration_soh,
            threshold_bias_band_soh=args.physics_threshold_bias_band_soh,
            threshold_bias_mode=args.physics_threshold_bias_mode,
            eol_event_phase_alignment=args.physics_eol_event_phase_alignment,
            eol_event_phase_clip_cycles=args.physics_eol_event_phase_clip_cycles,
        )
        payload = run_physics_mgi(
            args.data_dir, args.output_dir, config, test_batteries=args.test_names
        )
        print(format_physics_results(payload))
        print(f"\nSaved results to: {args.output_dir / 'results.json'}")
        return

    summary = load_cycle_summary(args.data_dir, args.dataset)
    frame = build_feature_frame(summary)
    if args.head == "residual-hgb":
        config = ResidualBoostingConfig(
            seq_len=args.seq_len,
            max_seq_len=args.max_seq_len,
            start_points=tuple(args.start_points),
            seed=args.seed,
            max_iter=args.hgb_max_iter,
            learning_rate=args.hgb_learning_rate,
            max_leaf_nodes=args.hgb_max_leaf_nodes,
            l2_regularization=args.hgb_l2,
            feature_mode=args.residual_feature_mode,
            rated_capacity=args.rated_capacity,
        )
        payload = run_residual_boosting(frame, config, args.output_dir, test_batteries=args.test_names)
        print(format_residual_results(payload))
        print(f"\nSaved results to: {args.output_dir / 'results.json'}")
        return

    config = TrainConfig(
        seq_len=args.seq_len,
        max_seq_len=args.max_seq_len,
        rated_capacity=args.rated_capacity,
        start_points=tuple(args.start_points),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        dropout=args.dropout,
        recon_weight=args.recon_weight,
        residual_anchor_weight=args.residual_anchor_weight,
        huber_beta=args.huber_beta,
    )
    payload = run_leave_one_battery_out(frame, config, args.output_dir, test_batteries=args.test_names)
    print(format_results(payload))
    print(f"\nSaved results to: {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
