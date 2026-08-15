from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(timestep: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class AdaLNZeroBlock(nn.Module):
    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dimension, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, 4 * dimension), nn.ReLU(),
            nn.Linear(4 * dimension, dimension),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dimension, 6 * dimension))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return value * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition).chunk(6, dim=1)
        normalized = self.modulate(self.norm1(value), shift1, scale1)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        value = value + gate1.unsqueeze(1) * attended
        normalized = self.modulate(self.norm2(value), shift2, scale2)
        return value + gate2.unsqueeze(1) * self.ffn(normalized)


class ConditionalDiT(nn.Module):
    """Equations (12)-(19) with paper Table 11 defaults."""

    def __init__(self, sequence_length: int, health_features: int = 12, patch_size: int = 4,
                 dimension: int = 256, heads: int = 8, depth: int = 12) -> None:
        super().__init__()
        if sequence_length % patch_size:
            raise ValueError("sequence_length must be divisible by patch_size")
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        tokens = sequence_length // patch_size
        self.noisy_embedding = nn.Linear(patch_size, dimension)
        self.history_embedding = nn.Linear(patch_size, dimension)
        self.health_embedding = nn.Linear(patch_size * health_features, dimension)
        self.position = nn.Parameter(torch.zeros(1, tokens, dimension))
        self.time_mlp = nn.Sequential(nn.Linear(dimension, dimension), nn.SiLU(),
                                      nn.Linear(dimension, dimension))
        self.soh_embedding = nn.Sequential(nn.Linear(1, dimension), nn.SiLU(),
                                           nn.Linear(dimension, dimension))
        self.blocks = nn.ModuleList([AdaLNZeroBlock(dimension, heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dimension, 2 * dimension))
        self.noise_head = nn.Linear(dimension, patch_size)
        self.variance_head = nn.Linear(dimension, patch_size)
        nn.init.normal_(self.position, std=0.02)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.noise_head.weight)
        nn.init.zeros_(self.noise_head.bias)

    def forward(self, noisy: torch.Tensor, timestep: torch.Tensor,
                health: torch.Tensor, soh_label: torch.Tensor,
                history: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch = noisy.shape[0]
        noisy_tokens = noisy.reshape(batch, -1, self.patch_size)
        health_tokens = health.reshape(batch, -1, self.patch_size * health.shape[-1])
        value = self.noisy_embedding(noisy_tokens) + self.health_embedding(health_tokens) + self.position
        if history is not None:
            value = value + self.history_embedding(
                history.reshape(batch, -1, self.patch_size)
            )
        condition = self.time_mlp(timestep_embedding(timestep, value.shape[-1]))
        condition = condition + self.soh_embedding(soh_label.reshape(batch, 1))
        for block in self.blocks:
            value = block(value, condition)
        shift, scale = self.final_modulation(condition).chunk(2, dim=1)
        value = self.final_norm(value) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        noise = self.noise_head(value).reshape(batch, self.sequence_length)
        variance = self.variance_head(value).reshape(batch, self.sequence_length)
        return noise, variance


class DiffusionSchedule(nn.Module):
    def __init__(self, steps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02) -> None:
        super().__init__()
        beta = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        posterior_variance = beta.clone()
        posterior_variance[1:] = beta[1:] * (1.0 - alpha_bar[:-1]) / (1.0 - alpha_bar[1:])
        posterior_variance[0] = 1e-20
        self.steps = steps
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))

    def add_noise(self, clean: torch.Tensor, timestep: torch.Tensor,
                  noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bar[timestep].unsqueeze(1)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    @torch.no_grad()
    def sample(self, model: ConditionalDiT, health: torch.Tensor,
               soh_label: torch.Tensor, generator: torch.Generator | None = None,
               known_prefix: torch.Tensor | None = None,
               history: torch.Tensor | None = None) -> torch.Tensor:
        value = torch.randn(
            health.shape[0], model.sequence_length, device=health.device, generator=generator
        )
        prefix_noise = (
            torch.randn(known_prefix.shape, device=value.device, dtype=value.dtype, generator=generator)
            if known_prefix is not None else None
        )
        for index in range(self.steps - 1, -1, -1):
            if known_prefix is not None:
                alpha_bar = self.alpha_bar[index]
                value[:, : known_prefix.shape[1]] = (
                    alpha_bar.sqrt() * known_prefix
                    + (1.0 - alpha_bar).sqrt() * prefix_noise
                )
            timestep = torch.full((len(value),), index, device=value.device, dtype=torch.long)
            predicted_noise, _ = model(value, timestep, health, soh_label, history)
            alpha = self.alpha[index]
            alpha_bar = self.alpha_bar[index]
            mean = (value - (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
            if index:
                noise = torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)
                value = mean + self.posterior_variance[index].sqrt() * noise
            else:
                value = mean
        if known_prefix is not None:
            value[:, : known_prefix.shape[1]] = known_prefix
        return value
