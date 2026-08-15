"""Non-anchored IC2ML adaptation for direct one-step capacity prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.IC2ML import CrossAttention, InceptionBlock, PositionalEncoding


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.context = args.context
        hidden = args.hidden_dim
        input_dim = getattr(args, "input_dim", 10)
        self.intra_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden // 2), nn.GELU(), nn.LayerNorm(hidden // 2),
            nn.Linear(hidden // 2, hidden), nn.LayerNorm(hidden),
        )
        self.use_cycle_input = bool(getattr(args, "use_cycle_input", False))
        self.use_capacity_history = bool(getattr(args, "use_capacity_history", False))
        self.cycle_embedding = (
            nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.LayerNorm(hidden))
            if self.use_cycle_input else None
        )
        self.input_norm = nn.LayerNorm(hidden)
        self.soh_predictor = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.LayerNorm(hidden // 2),
            nn.Linear(hidden // 2, 1),
        )
        self.intercycle_attention = nn.MultiheadAttention(
            hidden, num_heads=2, dropout=0.1, batch_first=True
        )
        self.pos_embedding = PositionalEncoding(hidden, dropout=0.1)
        self.attention_norm = nn.LayerNorm(hidden)
        self.flatten_processor = nn.Sequential(
            nn.Linear(self.context * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
        )
        self.image_encoder = nn.Sequential(
            InceptionBlock(1, 64), nn.MaxPool2d(kernel_size=2, stride=1),
            InceptionBlock(64, 128), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.image_projection = nn.Linear(128, hidden)
        self.cross_attention = CrossAttention(hidden)
        trajectory_width = hidden * 3 + 1 + (
            self.context if self.use_capacity_history else 0
        )
        self.trajectory_predictor = (
            nn.Linear(trajectory_width, 1)
            if self.use_capacity_history else
            nn.Sequential(
                nn.Linear(trajectory_width, hidden), nn.GELU(), nn.LayerNorm(hidden),
                nn.Linear(hidden, 1),
            )
        )
        self.rul_predictor = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, capacity_increment, absolute_cycles, observed_capacity_history=None):
        # IC2ML models aging from the sampled incremental-capacity curves.
        # Absolute cycle indices are intentionally not injected because their
        # cell-specific capacity relationship does not transfer reliably.
        embedded = self.intra_embedding(capacity_increment)
        if self.cycle_embedding is not None:
            embedded = embedded + self.cycle_embedding(absolute_cycles.unsqueeze(-1))
        embedded = self.input_norm(embedded)
        soh = 1.2 * torch.sigmoid(self.soh_predictor(embedded).squeeze(-1))
        positioned = self.pos_embedding(embedded)
        attended, _ = self.intercycle_attention(positioned, positioned, positioned)
        sequence = self.attention_norm(positioned + attended)
        context = self.flatten_processor(sequence.flatten(start_dim=1))

        image = self.image_encoder(capacity_increment.unsqueeze(1))
        image = self.image_projection(image.squeeze(-1).squeeze(-1))
        multimodal = self.cross_attention(context, image)
        # Couple the SOH and trajectory tasks explicitly: the state supplied
        # here is estimated from the native charge curve, never copied from a
        # measured previous-cycle capacity.
        feature_parts = [context, multimodal, sequence[:, -1], soh[:, -1:].contiguous()]
        if self.use_capacity_history:
            if observed_capacity_history is None:
                raise ValueError("observed capacity history is required by this variant")
            feature_parts.append(observed_capacity_history)
        features = torch.cat(feature_parts, dim=-1)
        trajectory_logits = self.trajectory_predictor(features)
        trajectory = (
            trajectory_logits
            if self.use_capacity_history else 1.2 * torch.sigmoid(trajectory_logits)
        )
        rul = self.rul_predictor(multimodal).squeeze(-1)
        return soh, trajectory, rul
