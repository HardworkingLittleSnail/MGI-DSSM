from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


STATE_NAMES = ("qli_ah", "cn_ah", "cp_ah", "r0_ohm", "rp_ohm")


def electrode_ocp_profile(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, 21)
    if name == "lco_graphite":
        un = np.asarray([
            1.20, .72, .45, .30, .22, .18, .155, .14, .13, .122, .116,
            .111, .106, .101, .096, .091, .086, .081, .075, .068, .06
        ])
        up = np.asarray([
            4.72, 4.58, 4.48, 4.40, 4.34, 4.29, 4.25, 4.21, 4.17, 4.13,
            4.09, 4.05, 4.01, 3.97, 3.93, 3.89, 3.84, 3.78, 3.70, 3.58, 3.35
        ])
    elif name == "nmc_graphite_siox":
        # Chen2020-style NMC/graphite-SiOx equilibrium-potential proxy. The
        # architecture and balance equations are unchanged; only chemistry's
        # fixed OCP lookup is selected per dataset.
        un = (
            1.9793 * np.exp(-39.3631 * x) + 0.2482
            - 0.0909 * np.tanh(29.8538 * (x - 0.1234))
            - 0.04478 * np.tanh(14.9159 * (x - 0.2769))
            - 0.0205 * np.tanh(30.4444 * (x - 0.6103))
        )
        up = (
            -0.8090 * x + 4.4875
            - 0.0428 * np.tanh(18.5138 * (x - 0.5542))
            - 17.7326 * np.tanh(15.7890 * (x - 0.3117))
            + 17.5842 * np.tanh(15.9308 * (x - 0.3120))
        )
    else:
        raise ValueError(f"Unknown electrode OCP profile: {name}")
    return x, un, up


class ElectrodeOCP(nn.Module):
    """Dataset-configured fixed electrode OCP prior."""

    def __init__(self, profile: str = "lco_graphite") -> None:
        super().__init__()
        x, un, up = electrode_ocp_profile(profile)
        self.register_buffer("x", torch.as_tensor(x, dtype=torch.float32))
        self.register_buffer("un", torch.as_tensor(un, dtype=torch.float32))
        self.register_buffer("up", torch.as_tensor(up, dtype=torch.float32))

    @staticmethod
    def _interp(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        pos = x.clamp(0.0, 1.0) * (values.numel() - 1)
        lo = pos.floor().long().clamp(0, values.numel() - 2)
        frac = pos - lo
        return values[lo] + frac * (values[lo + 1] - values[lo])

    def forward(self, xn: torch.Tensor, xp: torch.Tensor) -> torch.Tensor:
        return self._interp(xp, self.up) - self._interp(xn, self.un)


class MicroPhysicalSolver(nn.Module):
    """Electrode balance + one-RC voltage and absolute cutoff capacity."""

    def __init__(self, cutoff_voltage_v: float, discharge_current_a: float,
                 tau_p_seconds: float, q_grid_max_ah: float = 1.5,
                 q_grid_points: int = 400, xn_ref: float = .90,
                 xp_ref: float = .35, cutoff_sharpness_v: float = .008,
                 ocp_profile: str = "lco_graphite") -> None:
        super().__init__()
        self.ocp = ElectrodeOCP(ocp_profile)
        self.cutoff_voltage_v = float(cutoff_voltage_v)
        self.discharge_current_a = float(abs(discharge_current_a))
        self.tau_p_seconds = float(tau_p_seconds)
        self.xn_ref, self.xp_ref = float(xn_ref), float(xp_ref)
        self.cutoff_sharpness_v = float(cutoff_sharpness_v)
        self.register_buffer("q_grid", torch.linspace(0.0, q_grid_max_ah, q_grid_points))

    def initial_stoichiometry(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qli, cn, cp = z[..., 0], z[..., 1], z[..., 2]
        # Minimum-distance projection of shared healthy stoichiometries onto
        # the exact lithium-balance hyperplane cn*xn0 + cp*xp0 = qli.
        mismatch = qli - cn * self.xn_ref - cp * self.xp_ref
        denom = cn.square() + cp.square() + 1e-8
        xn0 = self.xn_ref + mismatch * cn / denom
        xp0 = self.xp_ref + mismatch * cp / denom
        return xn0, xp0

    def voltage_curve(self, z: torch.Tensor, q: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q_grid if q is None else q
        shape = [1] * (z.ndim - 1) + [q.numel()]
        qv = q.view(*shape)
        _, cn, cp, r0, rp = z.unbind(-1)
        xn0, xp0 = self.initial_stoichiometry(z)
        xn = xn0.unsqueeze(-1) - qv / cn.unsqueeze(-1).clamp_min(.2)
        xp = xp0.unsqueeze(-1) + qv / cp.unsqueeze(-1).clamp_min(.2)
        ocv = self.ocp(xn, xp)
        current = self.discharge_current_a
        time_s = 3600.0 * qv / current
        polarization = current * rp.unsqueeze(-1) * (
            1.0 - torch.exp(-time_s / self.tau_p_seconds)
        )
        return ocv - current * r0.unsqueeze(-1) - polarization

    def capacity(
        self, z: torch.Tensor, cutoff_voltage_v: torch.Tensor | float | None = None
    ) -> torch.Tensor:
        voltage = self.voltage_curve(z)
        # Discharge terminates at the first cutoff crossing. OCP clipping can
        # otherwise create a numerical rebound and incorrectly count capacity
        # after the cell has already reached 2.7 V.
        voltage = torch.cummin(voltage, dim=-1).values
        cutoff = self.cutoff_voltage_v if cutoff_voltage_v is None else cutoff_voltage_v
        cutoff = torch.as_tensor(cutoff, dtype=voltage.dtype, device=voltage.device)
        if cutoff.ndim:
            cutoff = cutoff.reshape(z.shape[:-1]).unsqueeze(-1)
        below = voltage <= cutoff
        has_crossing = below.any(dim=-1)
        first = below.to(torch.int64).argmax(dim=-1)
        first = torch.where(has_crossing, first, torch.full_like(first, voltage.shape[-1] - 1))
        upper = first.clamp(1, voltage.shape[-1] - 1)
        lower = upper - 1
        v0 = voltage.gather(-1, lower.unsqueeze(-1)).squeeze(-1)
        v1 = voltage.gather(-1, upper.unsqueeze(-1)).squeeze(-1)
        q0 = self.q_grid[lower]
        q1 = self.q_grid[upper]
        cutoff_value = cutoff.squeeze(-1) if cutoff.ndim else cutoff
        alpha = (cutoff_value - v0) / (v1 - v0).clamp(max=-1e-8)
        interpolated = q0 + alpha.clamp(0.0, 1.0) * (q1 - q0)
        return torch.where(has_crossing, interpolated, self.q_grid[-1].expand_as(interpolated))


class PhysicsGuidedStateModel(nn.Module):
    """Neural transition of internal states followed by physical capacity."""

    def __init__(self, cutoff_voltage_v: float, discharge_current_a: float,
                 tau_p_seconds: float, hidden_dim: int = 48,
                 num_layers: int = 1, dropout: float = 0.0,
                 state_mean: torch.Tensor | None = None,
                 state_std: torch.Tensor | None = None,
                 q_grid_max_ah: float = 1.5, q_grid_points: int = 400,
                 ocp_profile: str = "lco_graphite",
                 thermo_step_scale: float = 0.02,
                 kinetic_step_scale: float = 0.03,
                 trend_short_window: int = 8,
                 trend_long_window: int = 32) -> None:
        super().__init__()
        self.gru = nn.GRU(
            20, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
                                  nn.SiLU(), nn.Linear(hidden_dim, 5))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("state_mean", torch.zeros(5) if state_mean is None else state_mean.float())
        self.register_buffer("state_std", torch.ones(5) if state_std is None else state_std.float().clamp_min(1e-6))
        self.thermo_step_scale = float(thermo_step_scale)
        self.kinetic_step_scale = float(kinetic_step_scale)
        self.trend_short_window = int(trend_short_window)
        self.trend_long_window = int(trend_long_window)
        self.solver = MicroPhysicalSolver(
            cutoff_voltage_v, discharge_current_a, tau_p_seconds,
            q_grid_max_ah=q_grid_max_ah, q_grid_points=q_grid_points,
            ocp_profile=ocp_profile
        )

    def forward(
        self,
        history: torch.Tensor,
        cutoff_voltage_v: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        normalized = (history - self.state_mean) / self.state_std
        delta = F.pad(normalized[:, 1:] - normalized[:, :-1], (0, 0, 1, 0))

        def causal_average(values: torch.Tensor, window: int) -> torch.Tensor:
            channels = values.transpose(1, 2)
            padded = F.pad(channels, (window - 1, 0), mode="replicate")
            return F.avg_pool1d(padded, kernel_size=window, stride=1).transpose(1, 2)

        trend_short = causal_average(delta, self.trend_short_window)
        trend_long = causal_average(delta, self.trend_long_window)
        transition_input = torch.cat((normalized, delta, trend_short, trend_long), dim=-1)
        encoded, _ = self.gru(transition_input)
        temporal_context = encoded[:, -1]
        raw = self.head(temporal_context)
        last = history[:, -1]
        # Absolute next-state levels. Directionality remains a soft loss.
        thermo = last[:, :3] * torch.exp(self.thermo_step_scale * torch.tanh(raw[:, :3]))
        kinetic = last[:, 3:] * torch.exp(self.kinetic_step_scale * torch.tanh(raw[:, 3:]))
        z_next = torch.cat((thermo, kinetic), -1)
        return {
            "next_state": z_next,
            "capacity_ah": self.solver.capacity(z_next, cutoff_voltage_v),
            "voltage_curve": self.solver.voltage_curve(z_next),
        }
