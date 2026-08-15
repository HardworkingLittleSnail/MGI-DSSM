from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .config import ModelConfig


class RMSNorm(nn.Module):
    '''Equation (6), using the paper epsilon 1e-6.'''
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * x.square().mean(-1, keepdim=True).add(self.eps).rsqrt() * self.weight


class MultiScaleTokenizer(nn.Module):
    '''Non-overlapping multi-scale patch projection, equations (1)-(2).'''
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.lookback = cfg.lookback
        self.patch_lengths = cfg.patch_lengths
        self.projections = nn.ModuleList(
            nn.Linear(p * cfg.input_channels, cfg.d_model) for p in cfg.patch_lengths
        )
        self.scale_embeddings = (
            nn.Parameter(torch.zeros(len(cfg.patch_lengths), cfg.d_model))
            if cfg.use_scale_embeddings else None
        )
        if self.scale_embeddings is not None:
            nn.init.normal_(self.scale_embeddings, std=0.02)

    def forward(self, x: Tensor, input_mask: Tensor | None = None):
        if x.ndim != 3 or x.shape[1] != self.lookback:
            raise ValueError(f'Expected [B,{self.lookback},C], got {tuple(x.shape)}')
        batch, length, channels = x.shape
        if input_mask is None:
            input_mask = torch.ones(batch, length, dtype=torch.bool, device=x.device)
        else:
            input_mask = input_mask.to(dtype=torch.bool, device=x.device)
        tokens, positions, masks = [], [], []
        for i, (patch, projection) in enumerate(zip(self.patch_lengths, self.projections)):
            count = length // patch
            segments = x[:, :count * patch].reshape(batch, count, patch * channels)
            token = projection(segments)
            if self.scale_embeddings is not None:
                token = token + self.scale_embeddings[i]
            pos = torch.arange(count, device=x.device, dtype=x.dtype) * patch + (patch - 1) // 2
            valid = input_mask[:, :count * patch].reshape(batch, count, patch).all(-1)
            tokens.append(token)
            positions.append(pos.unsqueeze(0).expand(batch, -1))
            masks.append(valid)
        return tokens, positions, masks


class CrossScaleSE(nn.Module):
    '''Cross-scale squeeze-excitation attention, equations (3)-(4).'''
    def __init__(self, d_model: int, num_scales: int, reduction: int) -> None:
        super().__init__()
        reduced = max(1, d_model // reduction)
        self.fc1 = nn.Linear(d_model, reduced)
        self.fc2 = nn.Linear(reduced, num_scales * d_model)
        self.num_scales, self.d_model = num_scales, d_model

    def forward(self, branches: list[Tensor], masks: list[Tensor] | None = None):
        contexts = []
        for i, branch in enumerate(branches):
            if masks is None:
                contexts.append(branch.mean(1))
            else:
                weight = masks[i].unsqueeze(-1).to(branch.dtype)
                contexts.append((branch * weight).sum(1) / weight.sum(1).clamp_min(1))
        fused = torch.stack(contexts).sum(0)
        weights = torch.sigmoid(self.fc2(F.relu(self.fc1(fused))))
        weights = weights.reshape(-1, self.num_scales, self.d_model)
        return [branch * weights[:, i].unsqueeze(1) for i, branch in enumerate(branches)]


class CrossTimeReweighting(nn.Module):
    '''Grouped dual-branch CT operator reconstructed from Fig. 1 and Sec. III-C.

    The paper discloses its operations but no internal equations. This follows
    the shown 1x1 gating, local 3x3 branch and Softmax-MatMul fusion exactly.
    '''
    def __init__(self, d_model: int, groups: int, theta_init: float = 1.0) -> None:
        super().__init__()
        if d_model % groups:
            raise ValueError('d_model must be divisible by CT groups')
        channels = d_model // groups
        self.groups, self.channels_per_group = groups, channels
        self.context_conv = nn.Conv2d(channels, channels, 1)
        self.local_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.group_norm = nn.GroupNorm(channels, channels)
        # Equation (5) specifies gamma=softplus(theta) but not theta's initial
        # value.  Initializing the learnable scalar itself to one is the most
        # direct reconstruction and is retained as an explicit configuration.
        self.theta = nn.Parameter(torch.tensor(float(theta_init)))

    def forward(self, z: Tensor) -> Tensor:
        batch, time, hidden = z.shape
        grouped = z.transpose(1, 2).reshape(
            batch * self.groups, self.channels_per_group, time, 1
        )
        time_context = grouped.mean(3, keepdim=True)
        feature_context = grouped.mean(2, keepdim=True).transpose(2, 3)
        fused = self.context_conv(torch.cat((time_context, feature_context), 2))
        time_gate, feature_gate = torch.split(fused, (time, 1), 2)
        contextual = self.group_norm(
            grouped * time_gate.sigmoid() * feature_gate.transpose(2, 3).sigmoid()
        )
        local = self.local_conv(grouped)
        q_context = F.softmax(contextual.mean((2, 3)), -1).unsqueeze(1)
        q_local = F.softmax(local.mean((2, 3)), -1).unsqueeze(1)
        weights = q_context @ local.flatten(2) + q_local @ contextual.flatten(2)
        weights = weights.reshape(batch * self.groups, 1, time, 1).sigmoid()
        out = (grouped * weights).reshape(batch, hidden, time).transpose(1, 2)
        return F.softplus(self.theta) * out


def _apply_rope(x: Tensor, positions: Tensor, inv_freq: Tensor) -> Tensor:
    angles = positions.to(x.dtype).unsqueeze(-1) * inv_freq.to(x.dtype)
    sin, cos = angles.sin().unsqueeze(1), angles.cos().unsqueeze(1)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads, self.head_dim = num_heads, d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        inv = 1 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv, persistent=False)

    def forward(self, x: Tensor, positions: Tensor, valid_mask: Tensor) -> Tensor:
        batch, time, hidden = x.shape
        qkv = self.qkv(x).reshape(batch, time, 3, self.num_heads, self.head_dim)
        q, k, v = (item.transpose(1, 2) for item in qkv.unbind(2))
        q, k = _apply_rope(q, positions, self.inv_freq), _apply_rope(k, positions, self.inv_freq)
        scores = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
        scores = scores.masked_fill(
            ~valid_mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        attention = self.attn_dropout(scores.softmax(-1))
        output = (attention @ v).transpose(1, 2).reshape(batch, time, hidden)
        output = self.out(output) * valid_mask.unsqueeze(-1).to(output.dtype)
        return self.resid_dropout(output)


class GatedExpert(nn.Module):
    '''Up/Gate-SiLU-Hadamard-Down block shown in Fig. 1.'''
    def __init__(self, d_model: int, width: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, width, bias=False)
        self.gate = nn.Linear(d_model, width, bias=False)
        self.down = nn.Linear(width, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class SparseSharedMoE(nn.Module):
    '''Token-wise top-k routed experts plus shared expert, equations (7)-(8).'''
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        width = cfg.d_ff // cfg.top_k
        self.router = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList(
            GatedExpert(cfg.d_model, width, cfg.dropout) for _ in range(cfg.num_experts)
        )
        self.shared_expert = GatedExpert(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.shared_gate = nn.Linear(cfg.d_model, 1, bias=False)
        self.top_k, self.num_experts = cfg.top_k, cfg.num_experts

    def forward(self, x: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        probabilities = self.router(x).softmax(-1)
        top_prob, top_index = probabilities.topk(self.top_k, dim=-1)
        routed = torch.zeros_like(x)
        # Equation (8) uses original probabilities, not top-k-renormalized ones.
        for expert_index, expert in enumerate(self.experts):
            selected = top_index.eq(expert_index)
            token_selected = selected.any(-1) & valid_mask
            if token_selected.any():
                expert_output = expert(x[token_selected])
                weights = (top_prob * selected.to(top_prob.dtype)).sum(-1)[token_selected]
                routed[token_selected] += expert_output * weights.unsqueeze(-1)
        shared = torch.sigmoid(self.shared_gate(x)) * self.shared_expert(x)
        output = (routed + shared) * valid_mask.unsqueeze(-1).to(x.dtype)
        denom = valid_mask.sum().clamp_min(1).to(x.dtype)
        mean_probability = (probabilities * valid_mask.unsqueeze(-1)).sum((0, 1)) / denom
        uniform = torch.full_like(mean_probability, 1 / self.num_experts)
        auxiliary = (mean_probability - uniform).square().sum() / self.num_experts
        return output, auxiliary


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_epsilon)
        self.attention = RoPESelfAttention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.moe_norm = RMSNorm(cfg.d_model, cfg.rms_epsilon)
        self.moe = SparseSharedMoE(cfg)
        self.use_attention = cfg.use_attention
        self.use_moe = cfg.use_moe

    def forward(self, z: Tensor, positions: Tensor, mask: Tensor):
        if self.use_attention:
            z = z + self.attention(self.attn_norm(z), positions, mask)
        if self.use_moe:
            moe_output, auxiliary = self.moe(self.moe_norm(z), mask)
            z = z + moe_output
        else:
            auxiliary = z.new_zeros(())
        return z, auxiliary


@dataclass
class BATTERMoEOutput:
    prediction: Tensor
    auxiliary_loss: Tensor
    router_losses: tuple[Tensor, ...]


class BATTERMoE(nn.Module):
    '''BATTER-MoE equations (1)-(10), accepting [batch, lookback, channels].'''
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tokenizer = MultiScaleTokenizer(cfg)
        self.cross_scale = CrossScaleSE(cfg.d_model, len(cfg.patch_lengths), cfg.se_reduction)
        self.cross_time = CrossTimeReweighting(
            cfg.d_model, cfg.ct_groups, cfg.ct_theta_init
        )
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.num_layers))
        head_width = cfg.d_model + (
            cfg.input_channels if cfg.use_latest_observation_readout else 0
        )
        self.head = nn.Linear(head_width, 1)

    def forward(self, x: Tensor, input_mask: Tensor | None = None) -> BATTERMoEOutput:
        branches, positions, masks = self.tokenizer(x, input_mask)
        if self.cfg.use_cross_scale:
            branches = self.cross_scale(branches, masks)
        z = torch.cat(branches, 1)
        if self.cfg.use_cross_time:
            z = self.cross_time(z)
        position, valid = torch.cat(positions, 1), torch.cat(masks, 1)
        losses = []
        if self.cfg.use_encoder:
            for layer in self.layers:
                z, auxiliary = layer(z, position, valid)
                losses.append(auxiliary)
        masked_position = position.masked_fill(~valid, float('-inf'))
        latest = masked_position.max(1, keepdim=True).values
        selected = valid & position.eq(latest)
        pooled = (z * selected.unsqueeze(-1)).sum(1)
        pooled = pooled / selected.sum(1, keepdim=True).clamp_min(1)
        if self.cfg.use_latest_observation_readout:
            pooled = torch.cat((pooled, x[:, -1, :]), dim=-1)
        prediction = self.head(pooled).squeeze(-1)
        auxiliary = torch.stack(losses).mean() if losses else z.new_zeros(())
        return BATTERMoEOutput(prediction, auxiliary, tuple(losses))

    def loss(self, output: BATTERMoEOutput, target: Tensor):
        task = F.l1_loss(output.prediction, target)
        total = task + self.cfg.aux_loss_weight * output.auxiliary_loss
        return total, {'task_mae': task.detach(), 'auxiliary': output.auxiliary_loss.detach()}
