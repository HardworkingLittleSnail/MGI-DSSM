"""Build IC2ML's native 10-point charge inputs aligned to shared cleaned cycles."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dataloader import load_nasa_cycle_data
from prepare_calce_ic2ml import (
    _align, _charge_increment, _discharge_capacity, _first_timestamp, _number,
    iter_xlsx_rows,
)


def _summary(root: Path, dataset: str):
    names = {"nasa": "NASA", "calce": "CALCE", "tju": "TJU"}
    stem = names[dataset]
    filename = (
        "NASA_Data_minimal_interpolated.npy"
        if dataset == "nasa"
        else f"{stem}_Data_batter_moe_preprocessed.npy"
    )
    return np.load(
        root / "data" / "processed" / f"{stem} data" / filename,
        allow_pickle=True,
    ).item()


def _fill_causal(values: list[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray]:
    imputed = np.asarray([value is None for value in values], dtype=np.bool_)
    latest = None
    for index, value in enumerate(values):
        if value is not None and np.isfinite(value).all():
            latest = value
        elif latest is not None:
            values[index] = latest.copy()
    for index, value in enumerate(values):
        if value is None or not np.isfinite(value).all():
            values[index] = np.zeros(10, dtype=np.float32)
        else:
            break
    return np.stack(values).astype(np.float32), imputed


def build_nasa(
    root: Path, voltage_range: tuple[float, float], data_version: str = "processed"
):
    if data_version == "version3":
        source_root = root / "data" / "version3" / "NASA data"
        summary = np.load(
            source_root / "NASA_Data_minimal_interpolated.npy", allow_pickle=True
        )[0]
        result = {}
        for name in ("B0005", "B0006", "B0007", "B0018"):
            increments, raw_capacity, _ = load_nasa_cycle_data(
                source_root / f"{name}.mat", voltage_range=voltage_range
            )
            frame = summary[name].sort_values("Cycle").reset_index(drop=True)
            target = frame["Capacity"].to_numpy(dtype=np.float32)
            if len(increments) != len(target):
                source = [
                    {"discharge_capacity": float(c), "increment": x}
                    for c, x in zip(raw_capacity, increments)
                ]
                aligned, _ = _align(source, target)
                increments = np.stack([row["increment"] for row in aligned])
            result[name] = {
                "increments": increments.astype(np.float32),
                "capacities": target,
                "cycles": frame["Cycle"].to_numpy(dtype=np.int64),
                "imputed": np.zeros(len(target), dtype=np.bool_),
            }
        return result
    if data_version in ("version2.0", "version3"):
        result = {}
        version_root = "version3" if data_version == "version3" else "processed-version2.0"
        source_root = root / "data" / version_root / "NASA data"
        for name in ("B0005", "B0006", "B0007", "B0018"):
            increments, capacities, cycles = load_nasa_cycle_data(
                source_root / f"{name}.mat", voltage_range=voltage_range
            )
            result[name] = {
                "increments": increments.astype(np.float32),
                "capacities": capacities.astype(np.float32),
                "cycles": cycles.astype(np.int64),
                "imputed": np.zeros(len(capacities), dtype=np.bool_),
            }
        return result
    summary = _summary(root, "nasa")
    result = {}
    for name, frame in summary.items():
        increments, raw_capacity, _ = load_nasa_cycle_data(
            root / "data" / "raw" / "NASA data" / f"{name}.mat", voltage_range=voltage_range
        )
        target = frame["Capacity"].to_numpy(dtype=np.float32)
        if len(increments) != len(target):
            source = [{"discharge_capacity": float(c), "increment": x} for c, x in zip(raw_capacity, increments)]
            aligned, _ = _align(source, target)
            increments = np.stack([row["increment"] for row in aligned])
        result[name] = {"increments": increments.astype(np.float32), "capacities": target,
                        "cycles": frame["Cycle"].to_numpy(dtype=np.int64),
                        "imputed": np.zeros(len(target), dtype=np.bool_)}
    return result


def build_calce(
    root: Path, voltage_range: tuple[float, float], data_version: str = "processed"
):
    if data_version in ("version2.0", "version3"):
        version_root = "version3" if data_version == "version3" else "processed-version2.0"
        summary = np.load(
            root / "data" / version_root / "CALCE data" / "CALCE_Data.npy",
            allow_pickle=True,
        )[0]
        raw_root = root / "data" / version_root / "CALCE data"
    else:
        summary = _summary(root, "calce")
        raw_root = root / "data" / "raw" / "CALCE data"
    result = {}
    for battery, frame in summary.items():
        raw = []
        paths = sorted((raw_root / battery).glob("*.xlsx"), key=_first_timestamp)
        for path in paths:
            groups = {}
            for row in iter_xlsx_rows(path):
                cycle = _number(row, "Cycle_Index")
                if np.isfinite(cycle):
                    groups.setdefault(int(cycle), []).append(row)
            for cycle in sorted(groups):
                capacity = _discharge_capacity(groups[cycle])
                if capacity is not None:
                    raw.append({"discharge_capacity": capacity,
                                "increment": _charge_increment(groups[cycle], *voltage_range)})
        target = frame["Capacity"].to_numpy(dtype=np.float32)
        aligned, errors = _align(raw, target)
        increments, imputed = _fill_causal([row["increment"] for row in aligned])
        result[battery] = {"increments": increments, "capacities": target,
                           "cycles": frame["Cycle"].to_numpy(dtype=np.int64), "imputed": imputed}
        print(f"IC2ML CALCE {battery}: aligned={len(target)} mae={errors.mean():.8f}Ah", flush=True)
    return result


def _tju_increment(group: pd.DataFrame, voltage_range: tuple[float, float]):
    voltage = group["Ecell/V"].to_numpy(dtype=np.float64)
    current = group["<I>/mA"].to_numpy(dtype=np.float64)
    capacity = group["Q charge/mA.h"].to_numpy(dtype=np.float64) / 1000.0
    valid = np.isfinite(voltage) & np.isfinite(current) & np.isfinite(capacity) & (current > 20)
    voltage, capacity = voltage[valid], capacity[valid]
    if len(voltage) < 2 or voltage.min() > voltage_range[0] or voltage.max() < voltage_range[1]:
        return None
    order = np.argsort(voltage)
    voltage, capacity = voltage[order], capacity[order]
    voltage, unique = np.unique(voltage, return_index=True)
    sampled = np.interp(np.linspace(*voltage_range, 10), voltage, capacity[unique])
    return (sampled - sampled[0]).astype(np.float32)


def build_tju(
    root: Path, voltage_range: tuple[float, float], data_version: str = "processed"
):
    if data_version in ("version2.0", "version3"):
        version_root = "version3" if data_version == "version3" else "processed-version2.0"
        source = np.load(
            root / "data" / version_root / "TJU data"
            / "Dataset_3_NCM_NCA_battery_1C.npy",
            allow_pickle=True,
        )[0]
        mapping = {"CY25_1": "CY25-1", "CY25_2": "CY25-2", "CY25_3": "CY25-3"}
        indicator_columns = [
            "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
            "CC Q", "CC charge time", "voltage slope", "voltage entropy",
            "current mean", "current std", "current kurtosis", "current skewness",
            "CV Q", "CV charge time", "current slope", "current entropy",
        ]
        result = {}
        for source_name, name in mapping.items():
            frame = source[source_name].sort_values("Cycle").reset_index(drop=True)
            result[name] = {
                "increments": frame[indicator_columns].to_numpy(dtype=np.float32),
                "capacities": frame["Capacity"].to_numpy(dtype=np.float32),
                "cycles": frame["Cycle"].to_numpy(dtype=np.int64),
                "imputed": np.zeros(len(frame), dtype=np.bool_),
                "feature_names": indicator_columns,
            }
        return result
    summary = _summary(root, "tju")
    files = {"CY25-1": "CY25-05_1-#1.csv", "CY25-2": "CY25-05_1-#2.csv", "CY25-3": "CY25-05_1-#3.csv"}
    result = {}
    for name, filename in files.items():
        frame = summary[name]
        raw_cycles = frame["cycle index"].to_numpy(dtype=np.int64)
        raw = pd.read_csv(root / "data" / "raw" / "TJU data" / filename,
                          usecols=["Ecell/V", "<I>/mA", "Q charge/mA.h", "cycle number"])
        raw = raw[raw["cycle number"].isin(raw_cycles)]
        mapped = {int(c): _tju_increment(group, voltage_range) for c, group in raw.groupby("cycle number", sort=False)}
        increments, imputed = _fill_causal([mapped.get(int(c)) for c in raw_cycles])
        result[name] = {"increments": increments, "capacities": frame["Capacity"].to_numpy(dtype=np.float32),
                        "cycles": frame["Cycle"].to_numpy(dtype=np.int64), "imputed": imputed}
        print(f"IC2ML TJU {name}: cycles={len(frame)} imputed={int(imputed.sum())}", flush=True)
        del raw
    return result


def build(
    root: Path, dataset: str, voltage_range=(3.6, 3.7), force=False,
    data_version: str = "processed",
) -> Path:
    if data_version not in ("processed", "version2.0", "version3"):
        raise ValueError(f"unsupported data version: {data_version}")
    cache_version = (
        "processed_nasa_minimal_interpolated"
        if dataset == "nasa" and data_version == "processed"
        else data_version
    )
    cache_root = (
        root / "data" / "version3" / "native_inputs" / "ic2ml"
        if data_version == "version3"
        else root / "data" / "processed" / "native_inputs" / "ic2ml"
    )
    out = cache_root / cache_version
    out.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"{voltage_range[0]:g}-{voltage_range[1]:g}"
        if dataset != "tju" or data_version == "processed"
        else "16indicators"
    )
    path = out / f"{dataset}_{suffix}.npy"
    if path.exists() and not force:
        return path
    builders = {"nasa": build_nasa, "calce": build_calce, "tju": build_tju}
    payload = builders[dataset](root, voltage_range, data_version)
    np.save(path, payload, allow_pickle=True)
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("nasa", "calce", "tju"), required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(build(Path(__file__).resolve().parents[2], args.dataset, force=args.force))
