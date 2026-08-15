from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CALCE_SUMMARY = Path("CALCE data") / "CALCE_Data.npy"
SUMMARY_PATHS = {
    "calce": CALCE_SUMMARY,
    "nasa": Path("NASA data") / "NASA_Data.npy",
    "tju": Path("TJU data") / "TJU_Data.npy",
}


FEATURE_COLUMNS = [
    "capacity_norm",
    "cycle_norm",
    "log_cycle",
    "resistance",
    "ccct",
    "cvct",
    "d_resistance",
    "d_ccct",
    "d_cvct",
    "resistance_ma5",
    "ccct_ma5",
    "cvct_ma5",
]

THERMO_FEATURES = [
    "capacity_norm",
    "cycle_norm",
    "log_cycle",
    "ccct",
    "cvct",
    "d_ccct",
    "d_cvct",
    "ccct_ma5",
    "cvct_ma5",
]

KINETIC_FEATURES = [
    "cycle_norm",
    "log_cycle",
    "resistance",
    "d_resistance",
    "resistance_ma5",
    "cvct",
    "d_cvct",
]


TARGET_COLUMN = "capacity_norm"


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)


def load_calce_summary(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load the cycle-level CALCE summary distributed in this workspace."""

    path = data_dir / CALCE_SUMMARY
    if not path.exists():
        raise FileNotFoundError(f"CALCE summary not found: {path}")
    obj = np.load(path, allow_pickle=True)[0]
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected CALCE summary structure in {path}")

    out: Dict[str, pd.DataFrame] = {}
    required = {"BatteryName", "Cycle", "Capacity", "Resistance", "CCCT", "CVCT"}
    for name, frame in obj.items():
        df = frame.copy()
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        df = df.sort_values("Cycle").reset_index(drop=True)
        df["BatteryName"] = str(name)
        out[str(name)] = df
    return out


def load_cycle_summary(data_dir: Path, dataset: str = "calce") -> Dict[str, pd.DataFrame]:
    """Load CALCE, NASA or TJU cycle summaries into one common schema."""
    key = dataset.lower()
    if key not in SUMMARY_PATHS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if key == "calce":
        return load_calce_summary(data_dir)
    path = data_dir / SUMMARY_PATHS[key]
    if key == "tju" and not path.exists():
        from .raw_tju import prepare_tju_dataset

        prepare_tju_dataset(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{dataset} summary not found: {path}")
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape:
        obj = obj.item() if obj.shape == () else obj[0]
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected {dataset} summary structure in {path}")
    out: Dict[str, pd.DataFrame] = {}
    for name, frame in obj.items():
        df = frame.copy().sort_values("Cycle").reset_index(drop=True)
        df["BatteryName"] = str(name)
        # Map available charging summaries; unavailable indicators remain
        # explicitly missing and are causally imputed to zero downstream.
        if key == "tju":
            df["CCCT"] = pd.to_numeric(df.get("CC charge time"), errors="coerce")
            df["CVCT"] = pd.to_numeric(df.get("CV charge time"), errors="coerce")
            # The TJU protocol does not provide a cycle-wise resistance value.
            # MSTEA-Net instead uses constant-current discharge time (CCDT) as
            # its third health indicator. Reuse the common third-indicator slot
            # so the downstream sparse inverter receives CCCT/CVCT/CCDT.
            if "CC discharge time" in df:
                df["Resistance"] = pd.to_numeric(df["CC discharge time"], errors="coerce")
        for col in ["Resistance", "CCCT", "CVCT"]:
            if col not in df:
                df[col] = np.nan
        required = {"BatteryName", "Cycle", "Capacity"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        out[str(name)] = df
    return out


def build_feature_frame(summary: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build cycle-level macro-observation features without cross-cell leakage."""

    frames: List[pd.DataFrame] = []
    for name, src in summary.items():
        df = src[["BatteryName", "Cycle", "Capacity", "Resistance", "CCCT", "CVCT"]].copy()
        df["Cycle"] = df["Cycle"].astype(int)
        df = df.sort_values("Cycle").reset_index(drop=True)

        df["cycle_norm"] = (df["Cycle"].astype(float) - 1.0) / 1000.0
        df["log_cycle"] = np.log1p(df["Cycle"].astype(float)) / np.log1p(1000.0)
        for raw_col, feature_col in [
            ("Resistance", "resistance"),
            ("CCCT", "ccct"),
            ("CVCT", "cvct"),
        ]:
            values = pd.to_numeric(df[raw_col], errors="coerce")
            # Missing macro indicators are imputed only from earlier cycles of
            # the same cell. This avoids using held-out-cell or future values.
            df[feature_col] = values.ffill().fillna(0.0).astype(float)

        for col in ["resistance", "ccct", "cvct"]:
            df[f"d_{col}"] = df[col].diff().fillna(0.0)
            df[f"{col}_ma5"] = df[col].rolling(window=5, min_periods=1).mean()

        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.replace([np.inf, -np.inf], np.nan)
    checked_columns = [col for col in FEATURE_COLUMNS if col in merged.columns] + ["Capacity"]
    missing = merged[checked_columns].isna().sum()
    if int(missing.sum()) > 0:
        raise ValueError(f"Feature construction left missing values: {missing[missing > 0].to_dict()}")
    return merged


def capacity_bounds(
    frame: pd.DataFrame,
    train_batteries: Iterable[str],
    rated_capacity: float,
    max_seq_len: int | None = None,
) -> Tuple[float, float]:
    train_batteries = set(train_batteries)
    selected = frame[frame["BatteryName"].isin(train_batteries)].copy()
    if max_seq_len is not None:
        selected = selected[selected["Cycle"] <= int(max_seq_len)]
    values = selected["Capacity"].to_numpy(dtype=np.float32) / float(rated_capacity)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite capacity values for normalization.")
    lo = float(values.min())
    hi = float(values.max())
    if abs(hi - lo) < 1e-12:
        raise ValueError("Capacity normalization failed because training capacity is constant.")
    return lo, hi


def prepare_patchformer_frame(
    frame: pd.DataFrame,
    train_batteries: Iterable[str],
    rated_capacity: float = 1.1,
    max_seq_len: int | None = 1000,
) -> Tuple[pd.DataFrame, Tuple[float, float]]:
    """Prepare PatchFormer-style capacity normalization.

    The target/input capacity is first divided by rated capacity and then
    min-max normalized with training batteries only.
    """

    lo, hi = capacity_bounds(frame, train_batteries, rated_capacity, max_seq_len=max_seq_len)
    out = frame.copy()
    if max_seq_len is not None:
        out = out[out["Cycle"] <= int(max_seq_len)].copy()
    capacity = out["Capacity"].astype(float) / float(rated_capacity)
    out[TARGET_COLUMN] = (capacity - lo) / (hi - lo)
    return out, (lo, hi)


def battery_names(frame: pd.DataFrame) -> List[str]:
    return sorted(frame["BatteryName"].unique().tolist())


def feature_indices(columns: Iterable[str], selected: Iterable[str]) -> List[int]:
    col_list = list(columns)
    return [col_list.index(name) for name in selected]


def make_pair_arrays(
    frame: pd.DataFrame,
    train_batteries: Iterable[str],
    scaler: Standardizer | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Standardizer]:
    """Create one-step training pairs x_k -> x_{k+1}, Q_k -> Q_{k+1}."""

    train_batteries = set(train_batteries)
    train_frame = frame[frame["BatteryName"].isin(train_batteries)].copy()
    if train_frame.empty:
        raise ValueError("No training rows selected.")

    if scaler is None:
        scaler = Standardizer.fit(train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32))

    xs: List[np.ndarray] = []
    x_nexts: List[np.ndarray] = []
    us: List[np.ndarray] = []
    ys: List[float] = []
    y_nexts: List[float] = []

    for _, group in train_frame.groupby("BatteryName"):
        group = group.sort_values("Cycle").reset_index(drop=True)
        values = scaler.transform(group[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        caps = group["Capacity"].to_numpy(dtype=np.float32)
        cycles = group["Cycle"].to_numpy(dtype=np.float32)
        if len(group) < 2:
            continue
        xs.append(values[:-1])
        x_nexts.append(values[1:])
        ys.append(caps[:-1])
        y_nexts.append(caps[1:])
        u = np.stack(
            [
                (cycles[:-1] - 1.0) / 1000.0,
                np.ones_like(cycles[:-1]) / 1000.0,
            ],
            axis=1,
        ).astype(np.float32)
        us.append(u)

    if not xs:
        raise ValueError("No cycle pairs available for training.")

    return (
        np.concatenate(xs, axis=0).astype(np.float32),
        np.concatenate(x_nexts, axis=0).astype(np.float32),
        np.concatenate(us, axis=0).astype(np.float32),
        np.concatenate(ys, axis=0).astype(np.float32),
        np.concatenate(y_nexts, axis=0).astype(np.float32),
        scaler,
    )


class WindowStandardizer:
    def __init__(self, base: Standardizer) -> None:
        self.base = base

    @classmethod
    def fit(cls, frame: pd.DataFrame, train_batteries: Iterable[str]) -> "WindowStandardizer":
        train_batteries = set(train_batteries)
        selected = frame[frame["BatteryName"].isin(train_batteries)]
        return cls(Standardizer.fit(selected[FEATURE_COLUMNS].to_numpy(dtype=np.float32)))

    @property
    def mean(self) -> np.ndarray:
        return self.base.mean

    @property
    def std(self) -> np.ndarray:
        return self.base.std

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self.base.transform(values)


class PatchFormerWindowDataset(Dataset):
    """One-step sliding-window dataset: previous seq_len cycles -> next cycle."""

    def __init__(
        self,
        frame: pd.DataFrame,
        batteries: Sequence[str],
        scaler: WindowStandardizer,
        seq_len: int = 64,
        start_point: int | None = None,
        max_seq_len: int | None = 1000,
    ) -> None:
        self.frame = frame.copy()
        self.batteries = list(batteries)
        self.scaler = scaler
        self.seq_len = int(seq_len)
        self.start_point = start_point
        self.max_seq_len = max_seq_len
        self.samples: List[Tuple[str, int]] = []
        self.groups: Dict[str, pd.DataFrame] = {}

        for battery in self.batteries:
            group = self.frame[self.frame["BatteryName"] == battery].sort_values("Cycle").reset_index(drop=True)
            if max_seq_len is not None:
                group = group[group["Cycle"] <= int(max_seq_len)].reset_index(drop=True)
            if len(group) < self.seq_len + 1:
                continue
            self.groups[battery] = group
            for target_pos in range(self.seq_len, len(group)):
                cycle = int(group.iloc[target_pos]["Cycle"])
                if start_point is not None and cycle < int(start_point):
                    continue
                self.samples.append((battery, target_pos))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        battery, target_pos = self.samples[index]
        group = self.groups[battery]
        window = group.iloc[target_pos - self.seq_len : target_pos].copy()
        target = group.iloc[target_pos]
        x = self.scaler.transform(window[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        y = np.float32(target[TARGET_COLUMN])
        y_hist = window[TARGET_COLUMN].to_numpy(dtype=np.float32)
        cycle = int(target["Cycle"])
        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y, dtype=torch.float32),
            "y_hist": torch.from_numpy(y_hist),
            "cycle": torch.tensor(cycle, dtype=torch.long),
            "battery": battery,
        }


def denormalize_capacity_norm(value: np.ndarray | float, bounds: Tuple[float, float], rated_capacity: float) -> np.ndarray:
    lo, hi = bounds
    arr = np.asarray(value, dtype=np.float64)
    return (arr * (hi - lo) + lo) * float(rated_capacity)


def get_cycle_row(frame: pd.DataFrame, battery: str, cycle: int) -> pd.Series:
    rows = frame[(frame["BatteryName"] == battery) & (frame["Cycle"] == cycle)]
    if rows.empty:
        raise KeyError(f"{battery} cycle {cycle} not found.")
    return rows.iloc[0]


def transform_rows(rows: pd.DataFrame, scaler: Standardizer) -> np.ndarray:
    return scaler.transform(rows[FEATURE_COLUMNS].to_numpy(dtype=np.float32))


def linear_extrapolation(capacities: np.ndarray, window: int = 5) -> float:
    if len(capacities) == 0:
        raise ValueError("No capacities supplied.")
    if len(capacities) == 1:
        return float(capacities[-1])
    y = capacities[-window:]
    x = np.arange(len(y), dtype=np.float64)
    slope, intercept = np.polyfit(x, y.astype(np.float64), deg=1)
    return float(intercept + slope * len(y))
