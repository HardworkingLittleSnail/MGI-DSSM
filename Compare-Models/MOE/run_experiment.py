from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from batter_moe import BATTERMoE, get_paper_config
from batter_moe.data import prepare_data
from batter_moe.metrics import evaluate_from_start
from batter_moe.train import fit, predict, seed_everything


DEFAULT_ROOTS = {
    'nasa': Path('data/raw/NASA data'),
    'tju': Path('data/raw/TJU data'),
    'gotion': Path('data/raw/GOTION data'),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Reproduce BATTER-MoE held-out-cell experiments')
    parser.add_argument('--dataset', choices=tuple(DEFAULT_ROOTS), required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument('--tju-condition', default='CY25-05_1')
    parser.add_argument('--smoke-test', action='store_true', help='One epoch and one seed')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_paper_config(args.dataset)
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.smoke_test:
        config.max_epochs, args.seeds = 1, [args.seeds[0]]
    root = args.data_root or DEFAULT_ROOTS[args.dataset]
    prepared = prepare_data(config, root, args.tju_condition)
    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        seed_everything(seed)
        model = BATTERMoE(config.model)
        model, history = fit(
            model, prepared.train, prepared.validation, config, torch.device(args.device), seed
        )
        prediction, target = predict(
            model, prepared.test, config.batch_size, torch.device(args.device)
        )
        metrics = {
            str(sp): evaluate_from_start(
                prepared.test.target_cycles, target, prediction, sp, config.eol_fraction
            ) for sp in config.start_points
        }
        # The paper states that capacity is normalized as C/C0 but does not
        # state unambiguously whether Table IV restores Ah before MAE/RMSE.
        # Preserve the native normalized metrics and report the physical-unit
        # counterparts explicitly instead of silently choosing one convention.
        for item in metrics.values():
            item['mae_ah'] = item['mae'] * config.rated_capacity
            item['rmse_ah'] = item['rmse'] * config.rated_capacity
            item['capacity_metric_unit'] = 'normalized C/C0; *_ah fields are Ah'
        run = {'seed': seed, 'best_epoch': min(history, key=lambda x: x['validation_mae'])['epoch'], 'metrics': metrics}
        runs.append(run)
        torch.save(
            {'model': model.state_dict(), 'config': config.to_dict(), 'history': history, 'run': run},
            output_dir / f'seed_{seed}.pt'
        )
        np.savez(
            output_dir / f'predictions_seed_{seed}.npz',
            cycles=prepared.test.target_cycles, target=target, prediction=prediction
        )
        print(json.dumps(run, ensure_ascii=False))
    summary = {'config': config.to_dict(), 'runs': runs}
    for sp in config.start_points:
        for metric in ('mae', 'rmse', 'mae_ah', 'rmse_ah', 'r2', 're'):
            values = [run['metrics'][str(sp)][metric] for run in runs]
            summary.setdefault('aggregate', {}).setdefault(str(sp), {})[metric] = {
                'mean': float(np.mean(values)), 'std': float(np.std(values))
            }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )


if __name__ == '__main__':
    main()
