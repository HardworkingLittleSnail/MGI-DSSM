from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, Subset

from .config import ExperimentConfig


@dataclass
class CellSeries:
    cell_id: str
    features: np.ndarray
    capacity: np.ndarray
    cycles: np.ndarray | None = None


@dataclass
class MinMaxScaler:
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def fit(cls, arrays: list[np.ndarray]) -> 'MinMaxScaler':
        joined = np.concatenate(arrays, axis=0)
        return cls(np.nanmin(joined, 0), np.nanmax(joined, 0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        span = self.maximum - self.minimum
        span = np.where(span == 0, 1.0, span)
        return (values - self.minimum) / span


def clean_isolated_outliers(values: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    '''Remove only isolated 3-sigma deviations, then linearly interpolate.

    A deviation is measured from a local rolling median; sigma is estimated
    after trimming the largest residual decile so the anomaly cannot inflate
    its own threshold. This preserves multi-point regeneration runs.
    The paper does not specify its exact isolation detector; this explicit local
    definition is therefore an implementation assumption.
    '''
    frame = pd.DataFrame(np.asarray(values, dtype=np.float64).copy())
    for column in frame.columns:
        series = frame[column]
        baseline = series.rolling(5, center=True, min_periods=2).median()
        residual = series - baseline
        absolute = (residual - residual.median()).abs()
        core = residual[absolute <= absolute.quantile(0.9)]
        scale = core.std(skipna=True, ddof=0)
        threshold = sigma * (scale if np.isfinite(scale) else 0.0)
        candidate = absolute > threshold
        isolated = candidate & ~candidate.shift(1, fill_value=False) & ~candidate.shift(-1, fill_value=False)
        frame.loc[isolated, column] = np.nan
    return frame.interpolate(method='linear', limit_direction='both').to_numpy()


def load_nasa(root: str | Path) -> dict[str, CellSeries]:
    root = Path(root)
    cells = {}
    for path in sorted(root.glob('B*.mat')):
        cell_id = path.stem
        structure = loadmat(path, squeeze_me=True, struct_as_record=False)[cell_id]
        capacity = [float(c.data.Capacity) for c in np.atleast_1d(structure.cycle) if c.type == 'discharge']
        capacity = clean_isolated_outliers(np.asarray(capacity)[:, None])[:, 0]
        cells[cell_id] = CellSeries(cell_id, capacity[:, None], capacity)
    return cells


def load_tju(root: str | Path, condition: str = 'CY25-05_1') -> dict[str, CellSeries]:
    '''Load the paper CY25-1/2/3 split from one three-replicate TJU condition.'''
    paths = sorted(Path(root).rglob(f'{condition}-#*.csv'))
    if len(paths) != 3:
        raise FileNotFoundError(f'Expected three {condition} replicate CSVs under {root}, found {len(paths)}')
    cells = {}
    columns = [
        'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
        'voltage slope', 'voltage entropy', 'current mean', 'current std',
        'current kurtosis', 'current skewness', 'current slope', 'current entropy',
        'CC Q', 'CC charge time', 'CV Q', 'CV charge time', 'capacity'
    ]
    for path in paths:
        replicate = path.stem.rsplit('#', 1)[-1]
        cell_id = f'CY25-{replicate}'
        frame = pd.read_csv(path)
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f'{path} is missing paper indicators: {sorted(missing)}')
        # Preserve the feature grouping stated in Sec. IV-A rather than relying
        # on the incidental column order of a downloaded CSV.
        values = clean_isolated_outliers(frame[columns].to_numpy(dtype=np.float64))
        cells[cell_id] = CellSeries(cell_id, values, values[:, -1])
    return cells


def load_gotion(root: str | Path) -> dict[str, CellSeries]:
    '''Load Cell01/02/03 capacity trajectories when the external dataset is supplied.'''
    cells = {}
    for cell_id in ('Cell01', 'Cell02', 'Cell03'):
        matches = list(Path(root).rglob(f'{cell_id}*.csv'))
        if len(matches) != 1:
            raise FileNotFoundError(f'Expected one CSV for {cell_id}, found {len(matches)} under {root}')
        frame = pd.read_csv(matches[0])
        candidates = [c for c in frame.columns if c.lower() in ('capacity', 'discharge_capacity', 'q')]
        if not candidates:
            raise ValueError(f'Cannot identify capacity column in {matches[0]}')
        capacity = clean_isolated_outliers(frame[[candidates[0]]].to_numpy())[:, 0]
        cells[cell_id] = CellSeries(cell_id, capacity[:, None], capacity)
    return cells


def load_calce(root: str | Path) -> dict[str, CellSeries]:
    '''Load the four CALCE CS2 trajectories from the local cleaned cache.'''
    root = Path(root)
    path = root / 'CALCE_Data_clean_local3sigma_w21.npy'
    if not path.exists():
        path = root / 'CALCE_Data.npy'
    loaded = np.load(path, allow_pickle=True)
    payload = loaded.item() if loaded.shape == () else loaded[0]
    cells = {}
    for cell_id in ('CS2_35', 'CS2_36', 'CS2_37', 'CS2_38'):
        if cell_id not in payload:
            raise KeyError(f'{cell_id} missing from {path}')
        frame = payload[cell_id]
        capacity = frame['Capacity'].to_numpy(dtype=np.float64)
        # The preferred cache has already undergone the local 3-sigma/window-21
        # cleaning encoded in its filename; do not clean it a second time.
        if path.name == 'CALCE_Data.npy':
            capacity = clean_isolated_outliers(capacity[:, None])[:, 0]
        cells[cell_id] = CellSeries(cell_id, capacity[:, None], capacity)
    return cells


class WindowDataset(Dataset):
    '''One-step windows: cycles [k-L+1,k] predict capacity at k+1.'''
    def __init__(self, cells: list[CellSeries], lookback: int) -> None:
        windows, targets, cycles, cell_ids = [], [], [], []
        for cell in cells:
            for target_index in range(lookback, len(cell.capacity)):
                windows.append(cell.features[target_index - lookback:target_index])
                targets.append(cell.capacity[target_index])
                cycles.append(target_index + 1)  # one-based cycle number
                cell_ids.append(cell.cell_id)
        self.windows = torch.as_tensor(np.asarray(windows), dtype=torch.float32)
        self.targets = torch.as_tensor(np.asarray(targets), dtype=torch.float32)
        self.target_cycles = np.asarray(cycles, dtype=np.int64)
        self.cell_ids = np.asarray(cell_ids, dtype=object)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.windows[index], self.targets[index]


@dataclass
class PreparedData:
    train: Subset
    validation: Subset
    test: WindowDataset
    cells: dict[str, CellSeries]
    feature_scaler: MinMaxScaler | None


def normalize_cells(
    cells: dict[str, CellSeries], config: ExperimentConfig,
    scaler: MinMaxScaler | None = None,
) -> tuple[dict[str, CellSeries], MinMaxScaler | None]:
    '''Fit feature statistics on train/validation cells, never on held-out test.'''
    scaler = None
    if config.dataset == 'tju':
        # The instantaneous capacity is C/C0; the other 16 indicators use
        # training-only Min-Max statistics as stated in Sec. IV-A.
        capacity_index = 16
        non_capacity = [i for i in range(17) if i != capacity_index]
        if scaler is None:
            scaler = MinMaxScaler.fit([cells[c].features[:, non_capacity] for c in config.train_cells])

    normalized = {}
    for cell_id, cell in cells.items():
        if config.dataset == 'tju':
            features = cell.features.copy()
            features[:, non_capacity] = scaler.transform(features[:, non_capacity])
            features[:, capacity_index] = cell.capacity / config.rated_capacity
        else:
            features = cell.capacity[:, None] / config.rated_capacity
        capacity = cell.capacity / config.rated_capacity
        normalized[cell_id] = CellSeries(cell_id, features, capacity)
    return normalized, scaler


def prepare_data(
    config: ExperimentConfig,
    data_root: str | Path,
    tju_condition: str = 'CY25-05_1',
) -> PreparedData:
    if config.dataset == 'nasa':
        cells = load_nasa(data_root)
    elif config.dataset == 'tju':
        cells = load_tju(data_root, tju_condition)
    elif config.dataset == 'gotion':
        cells = load_gotion(data_root)
    elif config.dataset == 'calce':
        cells = load_calce(data_root)
    else:
        raise ValueError(config.dataset)
    missing = set(config.train_cells + (config.test_cell,)) - set(cells)
    if missing:
        raise KeyError(f'Missing cells required by paper split: {sorted(missing)}')
    # The paper states 80/20 but not chronological versus random. A fixed
    # shuffled split is the conventional interpretation and is reproducible.
    raw_pooled = WindowDataset([cells[c] for c in config.train_cells], config.model.lookback)
    generator = np.random.default_rng(config.split_seed)
    indices = generator.permutation(len(raw_pooled))
    split = int(round(len(indices) * (1 - config.validation_fraction)))
    scaler = None
    if config.dataset == 'tju':
        training_values = raw_pooled.windows[indices[:split], :, :16].numpy().reshape(-1, 16)
        scaler = MinMaxScaler.fit([training_values])
    cells, scaler = normalize_cells(cells, config, scaler)
    pooled = WindowDataset([cells[c] for c in config.train_cells], config.model.lookback)
    train, validation = Subset(pooled, indices[:split]), Subset(pooled, indices[split:])
    test = WindowDataset([cells[config.test_cell]], config.model.lookback)
    return PreparedData(train, validation, test, cells, scaler)
