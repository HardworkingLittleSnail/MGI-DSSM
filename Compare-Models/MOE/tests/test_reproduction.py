import unittest
import numpy as np
import torch

from batter_moe import BATTERMoE, get_paper_config
from batter_moe.data import clean_isolated_outliers
from batter_moe.metrics import evaluate_from_start
from batter_moe.model import MultiScaleTokenizer
from run_unified_benchmark import build_config, load_cells, prepare


class ModelTests(unittest.TestCase):
    def test_paper_config_shapes_and_rounded_parameter_counts(self):
        expected = {'nasa': '94K', 'gotion': '1.1M', 'tju': '5.4M'}
        for dataset in expected:
            cfg = get_paper_config(dataset)
            model = BATTERMoE(cfg.model)
            output = model(torch.randn(2, cfg.model.lookback, cfg.model.input_channels))
            self.assertEqual(output.prediction.shape, (2,))
            count = sum(p.numel() for p in model.parameters())
            if dataset == 'nasa':
                self.assertTrue(93000 < count < 96000)
            elif dataset == 'gotion':
                self.assertTrue(1.0e6 < count < 1.2e6)
            else:
                self.assertTrue(5.3e6 < count < 5.5e6)

    def test_token_counts_centers_and_patch_validity(self):
        cfg = get_paper_config('nasa').model
        tokenizer = MultiScaleTokenizer(cfg)
        x = torch.randn(1, 16, 1)
        mask = torch.ones(1, 16, dtype=torch.bool)
        mask[:, 3] = False
        tokens, positions, masks = tokenizer(x, mask)
        self.assertEqual([t.shape[1] for t in tokens], [8, 4, 2])
        self.assertEqual([p[0].tolist() for p in positions], [
            [0, 2, 4, 6, 8, 10, 12, 14], [1, 5, 9, 13], [3, 11]
        ])
        self.assertFalse(masks[0][0, 1])
        self.assertFalse(masks[1][0, 0])
        self.assertFalse(masks[2][0, 0])

    def test_load_balance_loss_is_zero_for_uniform_router(self):
        model = BATTERMoE(get_paper_config('nasa').model)
        with torch.no_grad():
            model.layers[0].moe.router.weight.zero_()
        output = model(torch.randn(3, 16, 1))
        self.assertAlmostEqual(output.auxiliary_loss.item(), 0.0, places=7)


class DataAndMetricTests(unittest.TestCase):
    def test_isolated_outlier_only(self):
        values = np.arange(12, dtype=float)[:, None]
        values[5] = 100
        cleaned = clean_isolated_outliers(values)
        self.assertAlmostEqual(cleaned[5, 0], 5.0)

    def test_local_data_splits_and_normalization(self):
        nasa = prepare(build_config('nasa', 'B0005', None), load_cells('nasa'))
        self.assertEqual((len(nasa[0]), len(nasa[1]), len(nasa[2])), (336, 84, 152))
        self.assertFalse(set(nasa[0].indices).intersection(nasa[1].indices))
        tju = prepare(build_config('tju', 'CY25-1', None), load_cells('tju'))
        self.assertEqual(tju[2].windows.shape[1:], (64, 17))
        self.assertEqual(set(tju[2].cell_ids), {'CY25-1'})

    def test_one_step_rul_protocol(self):
        cycles = np.arange(10, 16)
        target = np.array([.9, .8, .75, .69, .65, .6])
        prediction = np.array([.9, .8, .75, .72, .69, .6])
        result = evaluate_from_start(cycles, target, prediction, 10, .7)
        self.assertEqual(result['true_eol_cycle'], 13)
        self.assertEqual(result['predicted_eol_cycle'], 14)
        self.assertAlmostEqual(result['re'], 1 / 3)


if __name__ == '__main__':
    unittest.main()
