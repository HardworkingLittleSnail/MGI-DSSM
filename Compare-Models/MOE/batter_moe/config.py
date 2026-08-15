from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ModelConfig:
    input_channels: int
    lookback: int
    patch_lengths: tuple[int, ...]
    d_model: int
    num_layers: int
    d_ff: int
    num_experts: int
    top_k: int = 2
    dropout: float = 0.1
    aux_loss_weight: float = 1e-3
    num_heads: int = 8  # not disclosed by paper
    se_reduction: int = 16  # not disclosed by paper
    ct_groups: int = 8  # not disclosed by paper
    ct_theta_init: float = 1.0  # Eq. (5) defines theta but omits initialization
    use_scale_embeddings: bool = True  # paper says optional
    rms_epsilon: float = 1e-6
    # Explicit ablation switches. Defaults preserve the full paper model.
    use_cross_scale: bool = True
    use_cross_time: bool = True
    use_encoder: bool = True
    use_attention: bool = True
    use_moe: bool = True
    use_latest_observation_readout: bool = False

    def validate(self) -> None:
        if any(self.lookback % p != 0 for p in self.patch_lengths):
            raise ValueError('lookback must be divisible by every patch length')
        if self.d_model % self.num_heads or (self.d_model // self.num_heads) % 2:
            raise ValueError('d_model/num_heads must be an even integer for RoPE')
        if self.d_model % self.ct_groups:
            raise ValueError('d_model must be divisible by ct_groups')
        if not 1 <= self.top_k <= self.num_experts or self.d_ff % self.top_k:
            raise ValueError('invalid top_k')


@dataclass
class ExperimentConfig:
    dataset: str
    rated_capacity: float
    eol_fraction: float
    train_cells: tuple[str, ...]
    test_cell: str
    start_points: tuple[int, int]
    model: ModelConfig
    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 300
    patience: int = 10
    validation_fraction: float = 0.2
    split_seed: int = 42
    gradient_clip_norm: float = 0.0
    lr_plateau_factor: float = 1.0
    lr_plateau_patience: int = 5
    minimum_learning_rate: float = 1e-7

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_paper_config(dataset: str) -> ExperimentConfig:
    '''Return Tables I--III settings; undisclosed choices retain explicit defaults.'''
    name = dataset.lower()
    if name == 'nasa':
        model = ModelConfig(1, 16, (2, 4, 8), 64, 1, 128, 4, dropout=0.05)
        return ExperimentConfig(
            'nasa', 2.0, 0.70, ('B0006', 'B0007', 'B0018'), 'B0005', (50, 90), model
        )
    if name == 'gotion':
        model = ModelConfig(1, 64, (8, 16, 32), 128, 1, 512, 8, dropout=0.10)
        return ExperimentConfig(
            'gotion', 27.0, 0.80, ('Cell02', 'Cell03'), 'Cell01', (450, 750), model
        )
    if name == 'tju':
        model = ModelConfig(
            17, 64, (4, 8, 16), 256, 2, 1024, 4,
            dropout=0.10, aux_loss_weight=0.01
        )
        return ExperimentConfig(
            'tju', 2.5, 0.70, ('CY25-2', 'CY25-3'), 'CY25-1', (200, 400), model
        )
    raise ValueError(f'Unknown dataset {dataset!r}; expected nasa, gotion, or tju')


def get_calce_config(test_cell: str) -> ExperimentConfig:
    '''CALCE adaptation: compact paper NASA backbone with a 64-step window.'''
    cells = ('CS2_35', 'CS2_36', 'CS2_37', 'CS2_38')
    starts = {'CS2_35': 200, 'CS2_36': 200, 'CS2_37': 300, 'CS2_38': 300}
    if test_cell not in cells:
        raise ValueError(f'Unknown CALCE cell {test_cell!r}')
    # CALCE is not reported in the paper. Keep the compact NASA backbone and
    # adapt only lookback/patch scales to the requested 64-to-1 task.
    model = ModelConfig(1, 64, (8, 16, 32), 64, 1, 128, 4, dropout=0.05)
    return ExperimentConfig(
        'calce', 1.1, 0.80, tuple(c for c in cells if c != test_cell),
        test_cell, (starts[test_cell], starts[test_cell]), model
    )
