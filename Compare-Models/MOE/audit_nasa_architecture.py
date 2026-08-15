"""Audit paper-undisclosed BATTER-MoE choices on the NASA held-out-cell task.

This script is diagnostic: it never uses B0005 to choose a checkpoint.  Every
checkpoint is selected by validation MAE, exactly as in the paper protocol.
It reports both normalized C/C0 errors and errors converted to Ah so that the
unit ambiguity in the paper cannot silently affect comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from batter_moe import BATTERMoE, get_paper_config
from batter_moe.data import prepare_data
from batter_moe.metrics import evaluate_from_start
from batter_moe.train import fit, predict, seed_everything


VARIANTS = {
    "reconstructed_default": {},
    "scale_embedding_off": {"use_scale_embeddings": False},
    "heads_4": {"num_heads": 4},
    "heads_2": {"num_heads": 2},
    "se_reduction_8": {"se_reduction": 8},
    "se_reduction_4": {"se_reduction": 4},
    "ct_groups_4": {"ct_groups": 4},
    "ct_groups_16": {"ct_groups": 16},
    "ct_groups_32": {"ct_groups": 32},
    # Equation (5) defines gamma=softplus(theta), but the paper omits theta's
    # initialization.  These candidates test two common unit-scale choices.
    "ct_gamma_init_1": {"_ct_theta": 0.541324854612918},
    "ct_theta_init_0": {"_ct_theta": 0.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/NASA data"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/batter_moe_reproduction_audit/architecture_sweep"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 3, 8])
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS),
                        default=list(VARIANTS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows: list[dict[str, object]] = []

    def persist_runs() -> None:
        if not rows:
            return
        with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    for variant_name in args.variants:
        config = get_paper_config("nasa")
        changes = VARIANTS[variant_name]
        for key, value in changes.items():
            if not key.startswith("_"):
                setattr(config.model, key, value)
        config.model.validate()
        prepared = prepare_data(config, args.data_root)

        for seed in args.seeds:
            seed_everything(seed)
            model = BATTERMoE(config.model)
            if "_ct_theta" in changes:
                with torch.no_grad():
                    model.cross_time.theta.fill_(changes["_ct_theta"])
            model, history = fit(model, prepared.train, prepared.validation,
                                 config, device, seed)
            prediction, target = predict(model, prepared.test, config.batch_size, device)
            best = min(history, key=lambda item: item["validation_mae"])
            for start_point in config.start_points:
                metrics = evaluate_from_start(prepared.test.target_cycles, target, prediction,
                                              start_point, config.eol_fraction)
                row = {
                    "variant": variant_name,
                    "seed": seed,
                    "start_point": start_point,
                    "best_epoch": int(best["epoch"]),
                    "validation_mae_normalized": float(best["validation_mae"]),
                    "mae_normalized": metrics["mae"],
                    "rmse_normalized": metrics["rmse"],
                    "mae_ah": metrics["mae"] * config.rated_capacity,
                    "rmse_ah": metrics["rmse"] * config.rated_capacity,
                    "r2": metrics["r2"],
                    "re": metrics["re"],
                }
                rows.append(row)
            persist_runs()
            print(json.dumps(rows[-2:], ensure_ascii=False), flush=True)

    persist_runs()

    early = [row for row in rows if row["start_point"] == 50]
    summary = []
    for variant_name in args.variants:
        selected = [row for row in early if row["variant"] == variant_name]
        summary.append({
            "variant": variant_name,
            "runs": len(selected),
            "mean_mae_normalized": float(np.mean([row["mae_normalized"] for row in selected])),
            "std_mae_normalized": float(np.std([row["mae_normalized"] for row in selected])),
            "mean_mae_ah": float(np.mean([row["mae_ah"] for row in selected])),
            "mean_rmse_ah": float(np.mean([row["rmse_ah"] for row in selected])),
            "mean_re": float(np.mean([row["re"] for row in selected])),
            "mean_best_epoch": float(np.mean([row["best_epoch"] for row in selected])),
            "configuration": {**asdict(get_paper_config("nasa").model), **VARIANTS[variant_name]},
        })
    summary.sort(key=lambda item: item["mean_mae_normalized"])
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
