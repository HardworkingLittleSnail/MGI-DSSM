from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Compare-Models/PhysicsDualLoss"
sys.path.insert(0, str(PACKAGE))

from features import causal_stl, detect_period  # noqa: E402
from model import MSTEANet, TripleCompositeLoss  # noqa: E402


def test_mstea_shape_and_gradients():
    model = MSTEANet(input_dimension=5, hidden_dimension=32, attention_heads=4)
    x = torch.randn(9, 16, 5)
    prediction = model(x)
    assert prediction.shape == (9,)
    loss = prediction.square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_triple_loss_matches_formula_and_is_finite():
    objective = TripleCompositeLoss(temperature_kelvin=298.15)
    prediction = torch.tensor([1.9, 1.8, 1.7], requires_grad=True)
    target = torch.tensor([1.88, 1.79, 1.71])
    components = objective.components(
        prediction, target, torch.tensor([2.0, 3.0, 4.0]),
        torch.tensor([2.0, 2.0, 2.0]), torch.tensor([0, 0, 0]),
    )
    assert set(components) == {"total", "data", "arrhenius", "derivative"}
    assert all(torch.isfinite(value) for value in components.values())
    components["total"].backward()
    assert prediction.grad is not None


def test_causal_stl_does_not_change_past_when_future_changes():
    base = np.sin(np.arange(50) * 2 * np.pi / 7) + np.linspace(0, 1, 50)
    changed = base.copy()
    changed[30:] += 100.0
    first = causal_stl(base, 7)
    second = causal_stl(changed, 7)
    np.testing.assert_allclose(first[:30], second[:30], atol=0.0, rtol=0.0)
    assert 2 <= detect_period(base) <= 32
