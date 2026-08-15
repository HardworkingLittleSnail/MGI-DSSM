"""Targeted BATTER-MoE/TJU diagnostics using a single fixed seed.

This script never writes into benchmark result directories.  It first checks
whether the published-size network can memorize one fixed mini-batch; failure
of this test points to a forward/gradient/optimization defect rather than to
cross-cell generalization.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from batter_moe.model import BATTERMoE  # noqa: E402
from batter_moe.train import seed_everything  # noqa: E402
from run_unified_benchmark import build_config, load_cells, prepare  # noqa: E402


def gradient_norm(model: torch.nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().double().square().sum().cpu()
    return float(total.sqrt())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument(
        "--stage",
        choices=("patch", "cross-scale", "cross-time", "attention", "moe", "full"),
        default="full",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs" / "batter_moe_tju_diagnostics" / "seed_4_overfit",
    )
    args = parser.parse_args()

    seed = 4
    seed_everything(seed)
    config = build_config("tju", "CY25-1", None)
    stage_flags = {
        "patch": dict(use_cross_scale=False, use_cross_time=False, use_encoder=False),
        "cross-scale": dict(use_cross_scale=True, use_cross_time=False, use_encoder=False),
        "cross-time": dict(use_cross_scale=True, use_cross_time=True, use_encoder=False),
        "attention": dict(use_cross_scale=True, use_cross_time=True, use_encoder=True,
                          use_attention=True, use_moe=False),
        "moe": dict(use_cross_scale=True, use_cross_time=True, use_encoder=True,
                    use_attention=False, use_moe=True),
        "full": dict(use_cross_scale=True, use_cross_time=True, use_encoder=True,
                     use_attention=True, use_moe=True),
    }
    config = replace(config, model=replace(config.model, **stage_flags[args.stage]))
    if args.dropout is not None:
        config = replace(config, model=replace(config.model, dropout=args.dropout))
    cells = load_cells("tju")
    train, _, _, _ = prepare(config, cells)
    count = min(args.samples, len(train))
    x = torch.stack([train[i][0] for i in range(count)]).to(args.device)
    y = torch.stack([train[i][1] for i in range(count)]).to(args.device)

    # Seed immediately before construction so initialization is reproducible.
    seed_everything(seed)
    model = BATTERMoE(config.model).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rows: list[dict[str, float | int]] = []
    started = time.time()
    for step in range(args.steps + 1):
        model.train()
        output = model(x)
        loss, parts = model.loss(output, y)
        if step == args.steps:
            grad = float("nan")
        else:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = gradient_norm(model)
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            eval_output = model(x)
            eval_mae = torch.nn.functional.l1_loss(eval_output.prediction, y)
        row = {
            "step": step,
            "total_loss_normalized": float(loss.detach()),
            "mae_normalized": float(parts["task_mae"]),
            "mae_ah": float(parts["task_mae"] * config.rated_capacity),
            "eval_mae_ah": float(eval_mae * config.rated_capacity),
            "auxiliary": float(parts["auxiliary"]),
            "gradient_norm": grad,
            "elapsed_seconds": time.time() - started,
        }
        rows.append(row)
        if step == 0 or step % 10 == 0 or step == args.steps:
            print(
                f"step={step:03d} train_mae={row['mae_ah']:.8f} Ah "
                f"eval_mae={row['eval_mae_ah']:.8f} Ah "
                f"aux={row['auxiliary']:.8g} grad={grad:.5g} "
                f"elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "seed": seed,
        "samples": count,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "dropout": config.model.dropout,
        "gradient_clip": args.gradient_clip,
        "stage": args.stage,
        "device": args.device,
        "initial_mae_ah": rows[0]["mae_ah"],
        "final_mae_ah": rows[-1]["mae_ah"],
        "minimum_mae_ah": min(float(row["mae_ah"]) for row in rows),
        "final_eval_mae_ah": rows[-1]["eval_mae_ah"],
        "minimum_eval_mae_ah": min(float(row["eval_mae_ah"]) for row in rows),
        "elapsed_seconds": rows[-1]["elapsed_seconds"],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), args.output_dir / "checkpoint.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
