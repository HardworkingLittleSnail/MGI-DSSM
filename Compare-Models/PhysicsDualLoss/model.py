from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dimension: int, maximum_length: int = 2048) -> None:
        super().__init__()
        position = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dimension)
        )
        encoding = torch.zeros(maximum_length, dimension, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[: values.shape[1]].unsqueeze(0)


class MSTEANet(nn.Module):
    """Paper-faithful MSTEA-Net equations (11)-(28).

    Shape convention is batch x time x feature. The paper calls its attention
    block cross-variable attention, while equations (18)-(25) apply standard
    self-attention to the time-indexed hidden representations; this class
    follows those equations literally.
    """

    def __init__(self, input_dimension: int, hidden_dimension: int = 32,
                 attention_heads: int = 4) -> None:
        super().__init__()
        if hidden_dimension % attention_heads:
            raise ValueError("hidden_dimension must be divisible by attention_heads")
        self.input_projection = nn.Linear(input_dimension, hidden_dimension)
        self.lstm = nn.LSTM(hidden_dimension, hidden_dimension, batch_first=True)
        self.multiscale = nn.ModuleList([
            nn.Conv1d(hidden_dimension, hidden_dimension, kernel_size=kernel,
                      padding=kernel // 2)
            for kernel in (1, 3, 7)
        ])
        self.multiscale_projection = nn.Linear(3 * hidden_dimension, hidden_dimension)
        self.position = SinusoidalPositionEncoding(hidden_dimension)
        self.attention = nn.MultiheadAttention(
            hidden_dimension, attention_heads, batch_first=True, dropout=0.0
        )
        self.attention_norm = nn.LayerNorm(hidden_dimension)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dimension, 2 * hidden_dimension),
            nn.GELU(),
            nn.Linear(2 * hidden_dimension, hidden_dimension),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dimension)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dimension, 2 * hidden_dimension),
            nn.GELU(),
            nn.Linear(2 * hidden_dimension, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(values)
        recurrent, _ = self.lstm(projected)
        channels_first = recurrent.transpose(1, 2)
        scales = [F.gelu(convolution(channels_first)).transpose(1, 2)
                  for convolution in self.multiscale]
        fused = self.multiscale_projection(torch.cat(scales, dim=-1)) + recurrent
        positioned = self.position(fused)
        attended, _ = self.attention(positioned, positioned, positioned, need_weights=False)
        attended = self.attention_norm(positioned + attended)
        encoded = self.ffn_norm(attended + self.ffn(attended))
        return self.prediction_head(encoded[:, -1]).squeeze(-1)


class TripleCompositeLoss(nn.Module):
    """Equations (29)-(37): MSE plus static and differential Arrhenius losses."""

    def __init__(self, temperature_kelvin: float = 298.15, activation_energy_ev: float = 0.65,
                 exponent: float = 1.5, lambda_arr: float = 1e-4,
                 lambda_deriv: float = 1e-4) -> None:
        super().__init__()
        self.temperature_kelvin = float(temperature_kelvin)
        self.activation_energy_ev = float(activation_energy_ev)
        self.exponent = float(exponent)
        self.lambda_arr = float(lambda_arr)
        self.lambda_deriv = float(lambda_deriv)
        self.boltzmann_ev_per_kelvin = 8.617e-5

    @property
    def arrhenius_rate(self) -> float:
        return math.exp(
            -self.activation_energy_ev
            / (self.boltzmann_ev_per_kelvin * self.temperature_kelvin)
        )

    def components(self, prediction: torch.Tensor, target: torch.Tensor,
                   cycle: torch.Tensor, initial_capacity: torch.Tensor,
                   group: torch.Tensor) -> dict[str, torch.Tensor]:
        data = F.mse_loss(prediction, target)
        n = cycle.to(prediction.dtype)
        c0 = initial_capacity.to(prediction.dtype).clamp_min(1e-8)
        rate = prediction.new_tensor(self.arrhenius_rate)
        theoretical = 1.0 - torch.exp(-rate * n.pow(self.exponent))
        measured = (c0 - prediction) / c0
        arrhenius = F.mse_loss(measured, theoretical)

        # Formula (35) is evaluated only between truly consecutive predictions
        # from the same cell. The runner orders each full training epoch by cell
        # and cycle so no artificial cross-cell derivative is introduced.
        adjacent = (group[1:] == group[:-1]) & ((cycle[1:] - cycle[:-1]).abs() == 1)
        if bool(adjacent.any()):
            measured_derivative = measured[1:] - measured[:-1]
            theoretical_derivative = (
                rate * self.exponent * n[:-1].pow(self.exponent - 1.0)
                * torch.exp(-rate * n[:-1].pow(self.exponent))
            )
            derivative = F.mse_loss(
                measured_derivative[adjacent], theoretical_derivative[adjacent]
            )
        else:
            derivative = prediction.sum() * 0.0
        total = data + self.lambda_arr * arrhenius + self.lambda_deriv * derivative
        return {"total": total, "data": data, "arrhenius": arrhenius,
                "derivative": derivative}

    def forward(self, prediction: torch.Tensor, target: torch.Tensor,
                cycle: torch.Tensor, initial_capacity: torch.Tensor,
                group: torch.Tensor) -> torch.Tensor:
        return self.components(prediction, target, cycle, initial_capacity, group)["total"]
