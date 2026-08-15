import math

import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    """Inception block used by the authors' released IC2ML implementation."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 4
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(branch_channels),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(branch_channels),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm2d(branch_channels),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(branch_channels),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            dim=1,
        )


class CrossAttention(nn.Module):
    """Cross-attention block from the authors' released implementation."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = hidden_dim**-0.5

    def forward(self, context_features, cnn_features):
        query = self.query(context_features).unsqueeze(1)
        if cnn_features.ndim == 2:
            cnn_features = cnn_features.unsqueeze(1)
        key = self.key(cnn_features)
        value = self.value(cnn_features)
        scores = torch.bmm(query, key.transpose(1, 2)) * self.scale
        probabilities = self.softmax(scores)
        return torch.bmm(probabilities, value).squeeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[: x.size(1), :])


class Model(nn.Module):
    """Official IC2ML architecture corresponding to paper equations (1)-(10)."""

    def __init__(self, args):
        super().__init__()
        self.context = args.context
        self.horizon = args.horizon
        hidden_dim = args.hidden_dim
        input_dim = getattr(args, "input_dim", 10)
        self.use_capacity_history = bool(getattr(args, "use_capacity_history", False))

        self.intra_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.soh_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.intercycle_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=2,
            dropout=0.1,
            batch_first=True,
        )
        self.pos_embedding = PositionalEncoding(hidden_dim, dropout=0.1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        # Kept because it is part of the released model/state dict.
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.flatten_processor = nn.Sequential(
            nn.Linear(self.context * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.intraintercycle_embedding = nn.Sequential(
            InceptionBlock(1, 64),
            nn.MaxPool2d(kernel_size=2, stride=1),
            InceptionBlock(64, 128),
        )
        self.cnn_projection = nn.Linear(128, hidden_dim)
        self.trajectory_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.horizon),
        )
        self.history_embedding = (
            nn.Sequential(
                nn.Linear(self.context, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            if self.use_capacity_history else None
        )
        self.trajectory_fusion = (
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            if self.use_capacity_history else None
        )
        self.cross_attention = CrossAttention(hidden_dim)
        self.RUL_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, capacity_increment, start_volts=None, end_volts=None, tgt_soh=None,
        observed_capacity_history=None,
    ):
        if capacity_increment.ndim != 3:
            raise ValueError("capacity_increment must have shape [batch, cycles, 10]")
        if capacity_increment.size(1) != self.context:
            raise ValueError(
                f"expected context length {self.context}, got {tuple(capacity_increment.shape)}"
            )

        batch_size = capacity_increment.size(0)
        embedded = self.intra_embedding(capacity_increment)
        soh = self.soh_predictor(embedded).squeeze(-1)

        attended_input = self.pos_embedding(embedded)
        attention_output, _ = self.intercycle_attention(
            query=attended_input, key=attended_input, value=attended_input
        )
        # Paper Eq. (6): the residual branch is the original intra-cycle
        # embedding; positional encoding is used only by self-attention.
        intercycle = self.norm1(embedded + attention_output)
        context_features = self.flatten_processor(intercycle.reshape(batch_size, -1))

        cnn_features = self.intraintercycle_embedding(capacity_increment.unsqueeze(1))
        cnn_features = cnn_features.flatten(2).transpose(1, 2)
        cnn_features = self.cnn_projection(cnn_features)
        multimodal_features = self.cross_attention(context_features, cnn_features)

        trajectory_features = context_features
        if self.use_capacity_history:
            if observed_capacity_history is None:
                raise ValueError("observed capacity history is required")
            history_features = self.history_embedding(observed_capacity_history)
            trajectory_features = self.trajectory_fusion(
                torch.cat((context_features, history_features), dim=-1)
            )
        trajectory = self.trajectory_predictor(trajectory_features)
        rul = self.RUL_predictor(multimodal_features).squeeze(-1)
        return soh, trajectory, rul
