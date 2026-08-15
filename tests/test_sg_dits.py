from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Compare-Models/SG-DiTs"))

from model import ConditionalDiT, DiffusionSchedule  # noqa: E402


def test_sg_dits_shapes_and_gradient():
    model = ConditionalDiT(16, health_features=12, patch_size=4, dimension=32, heads=4, depth=2)
    diffusion = DiffusionSchedule(steps=20)
    clean = torch.randn(3, 16)
    noise = torch.randn_like(clean)
    timestep = torch.tensor([0, 5, 19])
    health = torch.randn(3, 16, 12)
    history = torch.randn(3, 16)
    soh = torch.rand(3)
    noisy = diffusion.add_noise(clean, timestep, noise)
    predicted, variance = model(noisy, timestep, health, soh, history)
    assert predicted.shape == variance.shape == clean.shape
    (predicted - noise).square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
