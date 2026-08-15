from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from batter_moe import BATTERMoE, get_calce_config
from batter_moe.data import prepare_data
from batter_moe.metrics import evaluate_from_start
from batter_moe.train import fit, predict, seed_everything


CELLS = ('CS2_35', 'CS2_36', 'CS2_37', 'CS2_38')
STARTS = {'CS2_35': 200, 'CS2_36': 200, 'CS2_37': 300, 'CS2_38': 300}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CALCE four-cell BATTER-MoE 64-to-1 experiment')
    parser.add_argument('--data-root', type=Path, default=Path('data/CALCE data'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/calce'))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-epochs', type=int)
    return parser.parse_args()


def aggregate(runs: list[dict]) -> dict:
    result = {}
    for group in ('normalized', 'ampere_hour'):
        result[group] = {}
        for metric in ('mae', 'rmse', 'r2', 're'):
            values = [run[group][metric] for run in runs]
            result[group][metric] = {
                'mean': float(np.mean(values)), 'std': float(np.std(values))
            }
    result['best_epoch'] = float(np.mean([run['best_epoch'] for run in runs]))
    result['stopping_epoch'] = float(np.mean([run['stopping_epoch'] for run in runs]))
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_runs, per_cell = [], {}
    for cell in CELLS:
        config = get_calce_config(cell)
        if args.max_epochs is not None:
            config.max_epochs = args.max_epochs
        prepared = prepare_data(config, args.data_root)
        cell_dir = args.output_dir / cell
        cell_dir.mkdir(parents=True, exist_ok=True)
        runs = []
        for seed in args.seeds:
            seed_everything(seed)
            model = BATTERMoE(config.model)
            model, history = fit(
                model, prepared.train, prepared.validation, config,
                torch.device(args.device), seed
            )
            prediction, target = predict(
                model, prepared.test, config.batch_size, torch.device(args.device)
            )
            metrics = evaluate_from_start(
                prepared.test.target_cycles, target, prediction,
                STARTS[cell], config.eol_fraction
            )
            normalized = {k: metrics[k] for k in ('mae', 'rmse', 'r2', 're')}
            ampere_hour = dict(normalized)
            ampere_hour['mae'] *= config.rated_capacity
            ampere_hour['rmse'] *= config.rated_capacity
            run = {
                'cell': cell, 'start_point': STARTS[cell], 'lookback': 64,
                'horizon': 1, 'seed': seed,
                'best_epoch': min(history, key=lambda row: row['validation_mae'])['epoch'],
                'stopping_epoch': len(history),
                'normalized': normalized, 'ampere_hour': ampere_hour,
                'rul': {k: metrics[k] for k in (
                    'true_eol_cycle', 'predicted_eol_cycle', 'true_rul', 'predicted_rul'
                )}
            }
            runs.append(run)
            all_runs.append(run)
            torch.save(
                {'model': model.state_dict(), 'config': config.to_dict(), 'history': history, 'run': run},
                cell_dir / f'seed_{seed}.pt'
            )
            np.savez(
                cell_dir / f'predictions_seed_{seed}.npz',
                cycles=prepared.test.target_cycles, target=target, prediction=prediction
            )
            print(json.dumps(run, ensure_ascii=False), flush=True)
        per_cell[cell] = {'runs': runs, 'aggregate': aggregate(runs)}
        (cell_dir / 'summary.json').write_text(
            json.dumps(per_cell[cell], ensure_ascii=False, indent=2), encoding='utf-8'
        )
    summary = {
        'protocol': {
            'cells': CELLS, 'start_points': STARTS, 'lookback': 64, 'horizon': 1,
            'seeds': args.seeds, 'split': 'leave-one-cell-out',
            'architecture': 'compact paper NASA backbone adapted to 64-to-1; patches 8/16/32'
        },
        'per_cell': per_cell,
        'overall_macro_40_runs': aggregate(all_runs)
    }
    (args.output_dir / 'summary_all_cells.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print('FINAL ' + json.dumps(summary['overall_macro_40_runs'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
