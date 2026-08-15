from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )
    last = net[-1]
    if isinstance(last, nn.Linear):
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, -2.5)
    return net


class SparseCycleInitializer(nn.Module):
    """Sparse macro-observation -> effective degradation proxy state."""

    def __init__(
        self,
        feature_dim: int,
        thermo_indices: Sequence[int],
        kinetic_indices: Sequence[int],
        hidden_dim: int = 48,
    ) -> None:
        super().__init__()
        self.register_buffer("thermo_indices", torch.tensor(list(thermo_indices), dtype=torch.long))
        self.register_buffer("kinetic_indices", torch.tensor(list(kinetic_indices), dtype=torch.long))
        self.thermo = _mlp(len(thermo_indices), hidden_dim, 3)
        self.kinetic = _mlp(len(kinetic_indices), hidden_dim, 3)
        self.feature_dim = int(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        thermo_x = flat.index_select(dim=1, index=self.thermo_indices)
        kinetic_x = flat.index_select(dim=1, index=self.kinetic_indices)
        z_thermo = F.softplus(self.thermo(thermo_x))
        z_kinetic = F.softplus(self.kinetic(kinetic_x))
        z = torch.cat([z_thermo, z_kinetic], dim=1)
        return z.reshape(*original_shape[:-1], 6)


class CapacitySolver(nn.Module):
    """Constrained capacity surrogate.

    Capacity is represented in PatchFormer-normalized units. Larger effective
    degradation states can only reduce the predicted capacity.
    """

    def __init__(self, state_dim: int = 6, initial_capacity_norm: float = 0.9) -> None:
        super().__init__()
        self.q_ref = nn.Parameter(torch.tensor(float(initial_capacity_norm)))
        self.raw_weights = nn.Parameter(torch.full((state_dim,), -4.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        weights = F.softplus(self.raw_weights)
        return self.q_ref - torch.sum(z * weights, dim=-1)


class MGIDSSMLite(nn.Module):
    """PatchFormer-protocol MGI-DSSM-lite.

    Input: previous 64 cycles, shape [B, seq_len, feature_dim].
    Output: next-cycle capacity in PatchFormer-normalized units.
    """

    def __init__(
        self,
        feature_dim: int,
        thermo_indices: Sequence[int],
        kinetic_indices: Sequence[int],
        hidden_dim: int = 64,
        initial_capacity_norm: float = 0.9,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.initializer = SparseCycleInitializer(feature_dim, thermo_indices, kinetic_indices, hidden_dim)
        self.temporal = nn.GRU(
            input_size=6,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 6),
            nn.Linear(hidden_dim + 6, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 6),
        )
        last = self.delta_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, -5.0)
        self.capacity_delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 6),
            nn.Linear(hidden_dim + 6, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        cap_last = self.capacity_delta_head[-1]
        if isinstance(cap_last, nn.Linear):
            nn.init.zeros_(cap_last.weight)
            nn.init.zeros_(cap_last.bias)
            cap_last.bias.data[0] = -7.0
        self.capacity = CapacitySolver(state_dim=6, initial_capacity_norm=initial_capacity_norm)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.initializer(x)

    def next_state(self, z_seq: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if context is None:
            _, h = self.temporal(z_seq)
            context = h[-1]
        last_z = z_seq[:, -1, :]
        raw = self.delta_head(torch.cat([context, last_z], dim=1))
        irreversible_delta = F.softplus(raw[:, :5]) * 0.02
        tau_delta = torch.tanh(raw[:, 5:6]) * 0.01
        return torch.cat(
            [
                last_z[:, :5] + irreversible_delta,
                torch.clamp(last_z[:, 5:6] + tau_delta, min=0.0),
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor, last_capacity: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z_seq = self.encode_sequence(x)
        _, h = self.temporal(z_seq)
        context = h[-1]
        z_next = self.next_state(z_seq, context=context)
        if last_capacity is None:
            pred = self.capacity(z_next)
            fade = torch.zeros_like(pred)
            reversible = torch.zeros_like(pred)
        else:
            raw_delta = self.capacity_delta_head(torch.cat([context, z_seq[:, -1, :]], dim=1))
            fade = F.softplus(raw_delta[:, 0]) * 0.05
            reversible = torch.tanh(raw_delta[:, 1]) * 0.01
            pred = last_capacity - fade + reversible
        recon = self.capacity(z_seq)
        return {
            "capacity": pred,
            "history_capacity": recon,
            "states": z_seq,
            "next_state": z_next,
            "fade_delta": fade,
            "reversible_delta": reversible,
        }
