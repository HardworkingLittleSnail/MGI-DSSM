from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.optimize import least_squares
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .metrics import patchformer_capacity_metrics
from .preprocessing import fit_train_minmax, isolated_sigma_interpolate
from .physics_model import (
    MicroPhysicalSolver,
    PhysicsGuidedStateModel,
    STATE_NAMES,
    electrode_ocp_profile,
)
from .raw_calce import iter_xlsx_rows


@dataclass
class PhysicsTrainConfig:
    dataset: str = "calce"
    seq_len: int = 64
    rated_capacity: float = 1.1
    start_points: tuple[int, ...] = (300, 400, 500)
    epochs: int = 60
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    lr_scheduler: str = "none"
    scheduler_min_lr_ratio: float = 0.05
    scheduler_patience: int = 6
    capacity_huber_beta: float = 0.01
    grad_clip_norm: float = 1.0
    hidden_dim: int = 48
    num_layers: int = 1
    dropout: float = 0.0
    seed: int = 7
    validation_fraction: float = 0.15
    early_stopping_patience: int = 12
    state_loss_weight: float = 0.4
    state_supervision: str = "coordinate"
    curve_loss_weight: float = 0.02
    weak_state_loss_weight: float = 0.02
    voltage_error_scale: float = 0.05
    capacity_loss_weight: float = 1.0
    regeneration_loss_weight: float = 0.0
    direction_loss_weight: float = 0.05
    late_life_weight: float = 0.5
    cutoff_voltage_v: float = 2.7
    discharge_current_a: float = 1.1
    tau_p_seconds: float = 120.0
    q_grid_max_ah: float = 1.5
    q_grid_points: int = 400
    max_self_reconstruction_mae: float = 0.008
    cache_name: str = "physics_curve_cache_v1.npz"
    summary_filename: str | None = None
    ocp_profile: str = "lco_graphite"
    outlier_sigma_window: int = 0
    outlier_preserve_endpoints: bool = False
    preprocessing_protocol: str = "legacy"
    state_scaling: str = "protocol"
    capacity_target_scaling: str = "protocol"
    thermo_step_scale: float = 0.02
    kinetic_step_scale: float = 0.03
    trend_short_window: int = 8
    trend_long_window: int = 32
    evaluation_protocol: str = "patchformer"
    threshold_bias_calibration_soh: float = 0.0
    threshold_bias_band_soh: float = 0.015
    threshold_bias_mode: str = "symmetric"
    eol_event_phase_alignment: str = "none"
    eol_event_phase_clip_cycles: int = 2


NASA_CUTOFF_VOLTAGE = {
    "B0005": 2.7,
    "B0006": 2.5,
    "B0007": 2.2,
    "B0018": 2.5,
}


def _dataset_dir(data_dir: Path, dataset: str) -> Path:
    directories = {"calce": "CALCE data", "nasa": "NASA data", "tju": "TJU data"}
    return data_dir / directories[dataset.lower()]


def _summary_path(
    data_dir: Path,
    dataset: str,
    summary_filename: str | None = None,
) -> Path:
    filenames = {"calce": "CALCE_Data.npy", "nasa": "NASA_Data.npy", "tju": "TJU_Data.npy"}
    filename = summary_filename or filenames[dataset.lower()]
    return _dataset_dir(data_dir, dataset) / filename


def _cutoff_for_battery(name: str, config: PhysicsTrainConfig) -> float:
    if config.dataset.lower() == "nasa":
        return NASA_CUTOFF_VOLTAGE[name]
    return config.cutoff_voltage_v


def _local_sigma_interpolate(
    values: np.ndarray, window: int, preserve_endpoints: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy local 3-sigma cleaner retained for old-result reproducibility."""
    y = np.asarray(values, dtype=np.float64).copy()
    flagged = np.zeros(len(y), dtype=bool)
    if window < 5:
        return y, flagged
    radius = window // 2
    for index, value in enumerate(y):
        if preserve_endpoints and index in {0, len(y) - 1}:
            continue
        left, right = max(0, index - radius), min(len(y), index + radius + 1)
        neighbours = np.r_[y[left:index], y[index + 1:right]]
        neighbours = neighbours[np.isfinite(neighbours)]
        if neighbours.size >= 4:
            mean, std = float(neighbours.mean()), float(neighbours.std())
            if std > 0 and abs(value - mean) > 3.0 * std:
                flagged[index] = True
    valid = np.flatnonzero(~flagged & np.isfinite(y))
    if flagged.any() and valid.size >= 2:
        y[flagged] = np.interp(np.flatnonzero(flagged), valid, y[valid])
    return y, flagged


def _configured_sigma_interpolate(
    values: np.ndarray, config: PhysicsTrainConfig
) -> tuple[np.ndarray, np.ndarray]:
    if config.outlier_sigma_window < 5:
        y = np.asarray(values, dtype=np.float64).copy()
        return y, np.zeros(len(y), dtype=bool)
    if config.preprocessing_protocol == "batter_moe":
        result = isolated_sigma_interpolate(
            values,
            window=config.outlier_sigma_window,
            sigma=3.0,
            min_neighbours=6,
            preserve_endpoints=True,
        )
        return result.values, result.repaired
    return _local_sigma_interpolate(
        values,
        config.outlier_sigma_window,
        config.outlier_preserve_endpoints,
    )


def _mstea_eol_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float
) -> dict[str, float]:
    """First-threshold-crossing errors with explicit right-censoring."""
    true_crossings = np.flatnonzero(np.asarray(y_true) <= threshold)
    pred_crossings = np.flatnonzero(np.asarray(y_pred) <= threshold)
    if not true_crossings.size or not pred_crossings.size:
        return {
            "rul_true": float("nan") if not true_crossings.size else float(true_crossings[0]),
            "rul_pred": float("nan") if not pred_crossings.size else float(pred_crossings[0]),
            "ae": float("nan"),
            "re": float("nan"),
        }
    true_index = int(true_crossings[0])
    pred_index = int(pred_crossings[0])
    ae = abs(pred_index - true_index)
    return {
        "rul_true": float(true_index),
        "rul_pred": float(pred_index),
        "ae": float(ae),
        "re": float(ae / max(true_index, 1)),
    }


def _threshold_local_bias_calibrate(
    pred: np.ndarray,
    rated_capacity: float,
    bias_soh: float,
    band_soh: float,
) -> np.ndarray:
    """Apply the retained symmetric EECC correction only near 70% SOH."""
    if bias_soh == 0.0:
        return np.asarray(pred, dtype=np.float64)
    rated = max(float(rated_capacity), 1e-6)
    soh = np.asarray(pred, dtype=np.float64) / rated
    weights = np.exp(-np.abs(soh - 0.7) / max(float(band_soh), 1e-6))
    return rated * (soh + float(bias_soh) * weights)


def _capacity_metrics_for_protocol(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: PhysicsTrainConfig,
) -> dict[str, float]:
    metrics = patchformer_capacity_metrics(y_true, y_pred, config.rated_capacity)
    if config.evaluation_protocol == "mstea":
        metrics.update(
            _mstea_eol_metrics(
                y_true,
                y_pred,
                config.rated_capacity * 0.7,
            )
        )
    return metrics


def _date(path: Path) -> date:
    m = re.search(r"_(\d{1,2})_(\d{1,2})_(\d{2})\.xlsx$", path.name)
    return date(2000 + int(m[3]), int(m[1]), int(m[2])) if m else date(1900, 1, 1)


def _first_timestamp(path: Path) -> float:
    """PatchFormer sorts workbooks by the first Date_Time cell, not filename."""
    try:
        first = next(iter(iter_xlsx_rows(path)))
        return float(first.get("Date_Time", ""))
    except (StopIteration, TypeError, ValueError):
        fallback = _date(path)
        return float(fallback.toordinal())


def _number(row: dict[str, str], key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _align_raw_curves(
    curves: list[dict[str, object]], target_capacity: np.ndarray
) -> tuple[list[dict[str, object]], float, float]:
    """Order-preserving minimum-cost mapping to PatchFormer's cleaned records."""
    raw_capacity = np.asarray([np.asarray(row["q"])[-1] for row in curves], dtype=np.float64)
    target = np.asarray(target_capacity, dtype=np.float64)
    n, m = len(raw_capacity), len(target)
    if n < m:
        raise ValueError(f"Cannot align {m} cleaned records to only {n} raw curves.")
    previous = np.full(m + 1, np.inf)
    previous[0] = 0.0
    take = np.zeros((n + 1, m + 1), dtype=np.bool_)
    for i in range(1, n + 1):
        current = np.full(m + 1, np.inf)
        current[0] = 0.0
        upper = min(i, m)
        for j in range(1, upper + 1):
            skip_cost = previous[j]
            match_cost = previous[j - 1] + (raw_capacity[i - 1] - target[j - 1]) ** 2
            if match_cost <= skip_cost:
                current[j] = match_cost
                take[i, j] = True
            else:
                current[j] = skip_cost
        previous = current
    indices: list[int] = []
    i, j = n, m
    while j > 0:
        if i <= 0:
            raise RuntimeError("Raw-to-PatchFormer alignment backtracking failed.")
        if take[i, j]:
            indices.append(i - 1)
            i, j = i - 1, j - 1
        else:
            i -= 1
    indices.reverse()
    errors = np.abs(raw_capacity[indices] - target)
    return [curves[index] for index in indices], float(errors.mean()), float(errors.max())


def _raw_nasa_cycles(
    data_dir: Path, batteries: list[str]
) -> dict[str, list[dict[str, object]]]:
    nasa_dir = data_dir / "NASA data"
    cache_path = nasa_dir / "raw_discharge_curves_physics_v1.npy"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True).item()
        if sorted(cached) == sorted(batteries):
            for battery in batteries:
                print(f"raw_cache battery={battery} cycles={len(cached[battery])} source=disk")
            return cached

    summary = np.load(nasa_dir / "NASA_Data.npy", allow_pickle=True)[0]
    output: dict[str, list[dict[str, object]]] = {}
    for battery in batteries:
        payload = loadmat(
            nasa_dir / f"{battery}.mat", squeeze_me=True, struct_as_record=False
        )
        records = np.atleast_1d(payload[battery].cycle)
        cutoff = NASA_CUTOFF_VOLTAGE[battery]
        latest_r0 = 0.05
        found: list[dict[str, object]] = []
        for record in records:
            operation = str(record.type).strip().lower()
            if operation == "impedance":
                resistance = np.asarray(record.data.Re, dtype=np.float64).reshape(-1)
                resistance = resistance[np.isfinite(resistance) & (resistance > 1e-6)]
                if resistance.size:
                    latest_r0 = float(np.median(resistance))
                continue
            if operation != "discharge":
                continue

            time = np.asarray(record.data.Time, dtype=np.float64).reshape(-1)
            voltage = np.asarray(record.data.Voltage_measured, dtype=np.float64).reshape(-1)
            signed_current = np.asarray(
                record.data.Current_measured, dtype=np.float64
            ).reshape(-1)
            size = min(len(time), len(voltage), len(signed_current))
            time, voltage, signed_current = (
                time[:size], voltage[:size], signed_current[:size]
            )
            valid = (
                np.isfinite(time)
                & np.isfinite(voltage)
                & np.isfinite(signed_current)
                & (signed_current < -1.0)
            )
            time, voltage, signed_current = (
                time[valid], voltage[valid], signed_current[valid]
            )
            if len(time) < 3:
                continue
            q = np.cumsum(-np.diff(time) * signed_current[1:] / 3600.0)
            voltage = voltage[1:]
            current = np.abs(signed_current[1:])
            crossing = np.flatnonzero(voltage <= cutoff)
            if not len(crossing):
                continue
            upper = int(crossing[0])
            if upper == 0:
                continue
            lower = upper - 1
            v0, v1 = voltage[lower], voltage[upper]
            fraction = np.clip((cutoff - v0) / min(v1 - v0, -1e-8), 0.0, 1.0)
            q_cutoff = q[lower] + fraction * (q[upper] - q[lower])
            q = np.r_[q[:upper], q_cutoff]
            voltage = np.r_[voltage[:upper], cutoff]
            current = np.r_[current[:upper], current[upper]]
            found.append(
                {
                    "q": q,
                    "v": voltage,
                    "current": float(np.median(current)),
                    "r0": latest_r0,
                }
            )

        target = summary[battery]["Capacity"].to_numpy(dtype=np.float64)
        output[battery], mean_error, max_error = _align_raw_curves(found, target)
        print(
            f"raw_cache battery={battery} raw_cycles={len(found)} "
            f"summary_cycles={len(output[battery])} "
            f"align_mae={mean_error:.8f}Ah align_max={max_error:.8f}Ah"
        )
    np.save(cache_path, output, allow_pickle=True)
    return output


def _raw_cycles(
    data_dir: Path,
    batteries: list[str],
    dataset: str = "calce",
    preprocessing_protocol: str = "legacy",
) -> dict[str, list[dict[str, object]]]:
    if dataset.lower() == "nasa":
        if preprocessing_protocol == "batter_moe":
            from .raw_nasa import prepare_nasa_raw_dataset

            _, curves = prepare_nasa_raw_dataset(data_dir)
            missing = sorted(set(batteries).difference(curves))
            if missing:
                raise ValueError(f"NASA raw curve cache missing batteries: {missing}")
            return {name: curves[name] for name in batteries}
        return _raw_nasa_cycles(data_dir, batteries)
    if dataset.lower() == "tju":
        from .raw_tju import prepare_tju_dataset

        version2_path = (
            Path(data_dir) / "TJU data" / "Dataset_3_NCM_NCA_battery_1C.npy"
        )
        if version2_path.exists():
            # version2.0 retains a cycle-indexed subset of the official TJU
            # records. Reuse the official discharge curves, then select by the
            # stored raw cycle indices instead of incorrectly taking a prefix.
            official_root = Path(data_dir).parent / "processed"
            _, official_curves = prepare_tju_dataset(official_root)
            payload = np.load(version2_path, allow_pickle=True)[0]
            mapping = {
                "CY25-1": "CY25_1", "CY25-2": "CY25_2", "CY25-3": "CY25_3"
            }
            curves = {}
            for name, source_name in mapping.items():
                indices = payload[source_name]["cycle index"].to_numpy(dtype=np.int64) - 1
                source_curves = official_curves[name]
                if np.any(indices < 0) or np.any(indices >= len(source_curves)):
                    raise ValueError(f"TJU version2 cycle index out of range: {name}")
                curves[name] = [source_curves[int(index)] for index in indices]
        else:
            _, curves = prepare_tju_dataset(data_dir)
        missing = sorted(set(batteries).difference(curves))
        if missing:
            raise ValueError(f"TJU curve cache missing batteries: {missing}")
        return {name: curves[name] for name in batteries}
    if preprocessing_protocol == "batter_moe":
        from .raw_calce import prepare_calce_raw_dataset

        _, curves = prepare_calce_raw_dataset(data_dir)
        missing = sorted(set(batteries).difference(curves))
        if missing:
            raise ValueError(f"CALCE raw curve cache missing batteries: {missing}")
        return {name: curves[name] for name in batteries}
    cache_path = data_dir / "CALCE data" / "raw_discharge_curves_patchformer_v4.npy"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True).item()
        if sorted(cached) == sorted(batteries):
            for battery in batteries:
                print(f"raw_cache battery={battery} cycles={len(cached[battery])} source=disk")
            return cached
    output: dict[str, list[dict[str, object]]] = {}
    summary = np.load(
        data_dir / "CALCE data" / "CALCE_Data.npy", allow_pickle=True
    )[0]
    for battery in batteries:
        found: list[dict[str, object]] = []
        paths = list((data_dir / "CALCE data" / battery).glob("*.xlsx"))
        for path in sorted(paths, key=_first_timestamp):
            groups: dict[int, list[dict[str, str]]] = {}
            for row in iter_xlsx_rows(path):
                try:
                    groups.setdefault(int(float(row["Cycle_Index"])), []).append(row)
                except (KeyError, ValueError):
                    continue
            for cycle in sorted(groups):
                # PatchFormer defines a discharge cycle by Step_Index == 7 and
                # integrates current over Test_Time. Use the identical rule.
                rows = [r for r in groups[cycle]
                        if int(_number(r, "Step_Index", -1)) == 7]
                if len(rows) < 2:
                    continue
                time = np.asarray([_number(r, "Test_Time(s)") for r in rows])
                v = np.asarray([_number(r, "Voltage(V)") for r in rows])
                signed_current = np.asarray([_number(r, "Current(A)") for r in rows])
                resistance = np.asarray([_number(r, "Internal_Resistance(Ohm)") for r in rows])
                valid = (np.isfinite(time) & np.isfinite(v) & np.isfinite(signed_current))
                time, v, signed_current, resistance = (
                    time[valid], v[valid], signed_current[valid], resistance[valid]
                )
                if len(time) < 2:
                    continue
                dq = -np.diff(time) * signed_current[1:] / 3600.0
                q = np.cumsum(dq)
                v, current, resistance = v[1:], np.abs(signed_current[1:]), resistance[1:]
                valid_q = np.isfinite(q) & (q >= 0)
                q, v, current, resistance = (
                    q[valid_q], v[valid_q], current[valid_q], resistance[valid_q]
                )
                if len(q) == 0:
                    continue
                rv = resistance[np.isfinite(resistance) & (resistance > 1e-6)]
                found.append({"q": q, "v": v, "current": float(np.median(current)),
                              "r0": float(np.median(rv)) if rv.size else 0.05})
        target = summary[battery]["Capacity"].to_numpy(dtype=np.float64)
        output[battery], mean_error, max_error = _align_raw_curves(found, target)
        print(
            f"raw_cache battery={battery} raw_cycles={len(found)} "
            f"patchformer_cycles={len(output[battery])} "
            f"align_mae={mean_error:.8f}Ah align_max={max_error:.8f}Ah"
        )
    np.save(cache_path, output, allow_pickle=True)
    return output


_X = np.linspace(0.0, 1.0, 21)
_UN = np.asarray([1.20, .72, .45, .30, .22, .18, .155, .14, .13, .122, .116,
                  .111, .106, .101, .096, .091, .086, .081, .075, .068, .06])
_UP = np.asarray([4.72, 4.58, 4.48, 4.40, 4.34, 4.29, 4.25, 4.21, 4.17, 4.13,
                  4.09, 4.05, 4.01, 3.97, 3.93, 3.89, 3.84, 3.78, 3.70, 3.58, 3.35])


def _fit(row: dict[str, object], tau_p_seconds: float,
         previous: np.ndarray | None = None,
         ocp_profile: str = "lco_graphite") -> np.ndarray:
    q, v = np.asarray(row["q"]), np.asarray(row["v"])
    ocp_x, ocp_un, ocp_up = electrode_ocp_profile(ocp_profile)
    observed_r0, current = float(row["r0"]), float(row["current"])
    qs = np.linspace(0, q[-1], min(96, len(q)))
    vs = np.interp(qs, q, v)

    def error(p: np.ndarray) -> np.ndarray:
        qli, cn, cp, r0 = p
        mismatch = qli - cn * .90 - cp * .35
        denom = cn * cn + cp * cp + 1e-8
        xn0 = .90 + mismatch * cn / denom
        xp0 = .35 + mismatch * cp / denom
        xn = np.clip(xn0 - qs / cn, 0.0, 1.0)
        xp = np.clip(xp0 + qs / cp, 0.0, 1.0)
        ocv = np.interp(xp, ocp_x, ocp_up) - np.interp(xn, ocp_x, ocp_un)
        time_s = 3600.0 * qs / max(current, 1e-3)
        polarization_factor = current * (1.0 - np.exp(-time_s / tau_p_seconds))
        rp_raw = (
            ocv[-1] - current * r0 - vs[-1]
        ) / max(polarization_factor[-1], 1e-6)
        rp = np.clip(rp_raw, .001, 1.20)
        predicted = ocv - current * r0 - polarization_factor * rp
        voltage_residual = predicted - vs
        # DVA provides electrode-specific shape information beyond the raw
        # voltage fit and helps separate QLi, Cn and Cp sensitivities.
        dva_pred = np.gradient(predicted, qs)
        dva_obs = np.gradient(vs, qs)
        dva_residual = .03 * np.clip(dva_pred - dva_obs, -2.0, 2.0)
        # Capacity is a cutoff event. Give the measured terminal point explicit
        # weight so a globally imperfect public OCP prior cannot fit the curve
        # while moving the physical cutoff to a different capacity.
        cutoff_residual = np.repeat(predicted[-1] - vs[-1], 20)
        # The equipment resistance field is an auxiliary observation, not a
        # hard R0 label because its pulse duration is undocumented.
        resistance_residual = np.asarray([(r0 - observed_r0) / .2])
        if previous is None:
            temporal_residual = np.empty(0)
        else:
            temporal_scale = np.asarray([.25, .35, .35, .04])
            temporal_residual = .015 * (p - previous[:4]) / temporal_scale
        rp_bound_residual = np.asarray([
            .1 * max(.001 - rp_raw, 0.0) + .1 * max(rp_raw - 1.20, 0.0)
        ])
        return np.r_[voltage_residual, dva_residual, cutoff_residual,
                     resistance_residual, temporal_residual, rp_bound_residual]

    initial_capacity = max(float(q[-1]), .7)
    qli_center = max(1.55, initial_capacity * 1.25)
    starts = [
        [qli_center, max(initial_capacity * 1.25, .25), max(initial_capacity * 1.25, .25),
         np.clip(observed_r0, .005, .2)],
        [max(1.40, initial_capacity * 1.10), max(initial_capacity * 1.05, .25), max(initial_capacity * 1.50, .25),
         np.clip(observed_r0, .005, .2)],
        [max(1.75, initial_capacity * 1.40), max(initial_capacity * 1.50, .25), max(initial_capacity * 1.05, .25),
         np.clip(observed_r0, .005, .2)],
    ]
    if previous is not None:
        starts[0] = previous[:4].tolist()
    # These are effective whole-cell parameters, not material constants.
    # Severe late-life CALCE cycles require a wider domain than healthy-cell
    # priors; physical validity is enforced by stoichiometry and cutoff losses.
    lower = np.asarray([.20, .15, .15, .001], dtype=np.float64)
    # Scale the effective-capacity domain with the cell instead of imposing
    # CALCE/NASA-sized bounds on higher-capacity TJU cells.
    capacity_upper = max(3.50, initial_capacity * 2.0)
    upper = np.asarray([max(2.60, initial_capacity * 2.0),
                        capacity_upper, capacity_upper, .60], dtype=np.float64)
    # scipy requires x0 to lie strictly inside the feasible box. This also
    # handles rare malformed/high-capacity cycles and round-off at a bound.
    feasible_starts = [
        np.clip(np.asarray(start, dtype=np.float64), lower + 1e-6, upper - 1e-6)
        for start in starts
    ]
    solutions = [
        least_squares(error, start,
                      bounds=(lower, upper),
                      loss="soft_l1", f_scale=.01, max_nfev=80)
        for start in feasible_starts
    ]
    best = min(solutions, key=lambda item: item.cost)
    qli, cn, cp, r0 = best.x
    mismatch = qli - cn * .90 - cp * .35
    denom = cn * cn + cp * cp + 1e-8
    xn0 = .90 + mismatch * cn / denom
    xp0 = .35 + mismatch * cp / denom
    q_end = float(q[-1])
    xn_end = np.clip(xn0 - q_end / cn, 0.0, 1.0)
    xp_end = np.clip(xp0 + q_end / cp, 0.0, 1.0)
    ocv_end = np.interp(xp_end, ocp_x, ocp_up) - np.interp(xn_end, ocp_x, ocp_un)
    t_end = 3600.0 * q_end / max(current, 1e-3)
    factor_end = current * (1.0 - np.exp(-t_end / tau_p_seconds))
    rp = np.clip(
        (ocv_end - current * r0 - float(v[-1])) / max(factor_end, 1e-6),
        .001, 1.20
    )
    return np.asarray([qli, cn, cp, r0, rp], np.float32)


def load_states(data_dir: Path, batteries: list[str], config: PhysicsTrainConfig) -> dict[str, object]:
    base = Path(config.cache_name)
    dataset_dir = _dataset_dir(data_dir, config.dataset)
    cache = dataset_dir / f"{base.stem}_micro5{base.suffix}"
    if cache.exists():
        p = np.load(cache, allow_pickle=True)
        if sorted(p["names"].tolist()) == sorted(batteries):
            return {
                "states": p["states"].item(),
                "capacities": p["capacities"].item(),
                "cutoffs": {name: _cutoff_for_battery(name, config) for name in batteries},
            }
    raw = _raw_cycles(
        data_dir, batteries, config.dataset, config.preprocessing_protocol
    )
    summary_obj = np.load(
        _summary_path(data_dir, config.dataset, config.summary_filename),
        allow_pickle=True,
    )[0]
    for name in batteries:
        expected = len(summary_obj[name])
        official_capacity = summary_obj[name]["Capacity"].to_numpy(dtype=np.float64)
        if len(raw[name]) > expected:
            raw[name], align_mae, align_max = _align_raw_curves(
                raw[name], official_capacity
            )
            print(
                f"raw_summary_align battery={name} raw_selected={len(raw[name])} "
                f"summary={expected} mae={align_mae:.8f}Ah max={align_max:.8f}Ah"
            )
        if len(raw[name]) != expected:
            raise ValueError(
                f"Raw-curve alignment failed for {name}: "
                f"raw-filtered={len(raw[name])}, summary={expected}"
            )
        official_capacity, flagged = _configured_sigma_interpolate(official_capacity, config)
        if config.outlier_sigma_window >= 5:
            print(
                f"mstea_preprocess battery={name} window={config.outlier_sigma_window} "
                f"interpolated={int(flagged.sum())}"
            )
        # Current-integrated endpoints and cycle-summary capacity can differ
        # slightly. Put each voltage curve on the cleaned capacity axis so the
        # inversion target and evaluation label share one physical q coordinate.
        for index, row in enumerate(raw[name]):
            q = np.asarray(row["q"], dtype=np.float64)
            if q.size == 0 or q[-1] <= 0:
                raise ValueError(f"Invalid capacity axis for {name} cycle {index + 1}.")
            aligned = dict(row)
            aligned["q"] = q * (official_capacity[index] / q[-1])
            raw[name][index] = aligned
    states: dict[str, np.ndarray] = {}
    for name in batteries:
        fitted: list[np.ndarray] = []
        previous = None
        for index, row in enumerate(raw[name], start=1):
            previous = _fit(row, config.tau_p_seconds, previous, config.ocp_profile)
            fitted.append(previous)
            if index % 100 == 0 or index == len(raw[name]):
                print(f"physics_inverse battery={name} cycles={index}/{len(raw[name])}")
        states[name] = np.stack(fitted)
    # Evaluation labels use the selected cleaned capacity summary rather than
    # independently reconstructed raw endpoints.
    capacities = {}
    for name in batteries:
        capacity, _ = _configured_sigma_interpolate(
            summary_obj[name]["Capacity"].to_numpy(dtype=np.float64), config
        )
        capacities[name] = capacity.astype(np.float32)
    np.savez_compressed(cache, names=np.asarray(batteries),
                        states=np.asarray(states, object), capacities=np.asarray(capacities, object))
    return {
        "states": states,
        "capacities": capacities,
        "cutoffs": {name: _cutoff_for_battery(name, config) for name in batteries},
    }


class Windows(Dataset):
    def __init__(self, states: dict[str, np.ndarray], capacities: dict[str, np.ndarray],
                 cutoffs: dict[str, float],
                 names: list[str], seq_len: int, start: int | None = None,
                 tail: float | None = None) -> None:
        self.states, self.capacities = states, capacities
        self.cutoffs, self.seq_len = cutoffs, seq_len
        self.samples: list[tuple[str, int]] = []
        for name in names:
            n = min(len(states[name]), len(capacities[name]))
            split = int(n * (1 - tail)) if tail is not None else n
            begin, end = (max(seq_len, split), n) if tail is not None else (seq_len, n)
            self.samples += [(name, i) for i in range(begin, end)
                             if start is None or i + 1 >= start]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        name, i = self.samples[index]
        return {"states": torch.from_numpy(self.states[name][i-self.seq_len:i]).float(),
                "next_state": torch.from_numpy(self.states[name][i]).float(),
                "capacity": torch.tensor(self.capacities[name][i]).float(),
                "last_capacity": torch.tensor(self.capacities[name][i-1]).float(),
                "cutoff_voltage_v": torch.tensor(self.cutoffs[name]).float(),
                "cycle": i + 1, "battery": name}


def _scores(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
    mae = float(np.mean(abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    den = np.sum((true - true.mean()) ** 2)
    return mae, rmse, float(1 - np.sum((pred - true) ** 2) / den) if den else 0.


@torch.no_grad()
def _evaluate(
    model: PhysicsGuidedStateModel,
    loader: DataLoader,
    config: PhysicsTrainConfig,
    apply_eecc: bool = True,
):
    model.eval()
    rows = []
    for batch in loader:
        raw_p = model(
            batch["states"], cutoff_voltage_v=batch["cutoff_voltage_v"]
        )["capacity_ah"].numpy()
        y = batch["capacity"].numpy()
        for j in range(len(raw_p)):
            rows.append({"battery": batch["battery"][j], "cycle": int(batch["cycle"][j]),
                         "true_ah": float(y[j]),
                         "raw_pred_ah": float(raw_p[j]),
                         "pred_ah": float(raw_p[j]),
                         "eecc_delta_ah": 0.0,
                         "eecc_beta_soh": float(config.threshold_bias_calibration_soh),
                         "persistence_ah": float(batch["last_capacity"][j])})
    if apply_eecc and rows and config.threshold_bias_calibration_soh != 0.0:
        raw = np.asarray([float(row["raw_pred_ah"]) for row in rows], dtype=np.float64)
        calibrated = _threshold_local_bias_calibrate(
            raw,
            config.rated_capacity,
            config.threshold_bias_calibration_soh,
            config.threshold_bias_band_soh,
        )
        for row, value, raw_value in zip(rows, calibrated, raw):
            row["pred_ah"] = float(value)
            row["eecc_delta_ah"] = float(value - raw_value)
    pred = np.asarray([float(row["pred_ah"]) for row in rows], dtype=np.float64)
    true = np.asarray([float(row["true_ah"]) for row in rows], dtype=np.float64)
    return (*_scores(pred, true), rows)


@torch.no_grad()
def _estimate_eol_event_phase_delta(
    model: PhysicsGuidedStateModel,
    data: dict[str, object],
    train_names: list[str],
    config: PhysicsTrainConfig,
) -> tuple[int, dict[str, object]]:
    """Estimate event latency from training batteries only."""
    mode = str(config.eol_event_phase_alignment).lower()
    if mode in {"", "none"}:
        return 0, {"mode": "none", "num_phase_batteries": 0}
    if mode not in {"global", "robust"}:
        raise ValueError("eol_event_phase_alignment must be one of: none, global, robust")

    start = int(config.start_points[0]) if config.start_points else config.seq_len
    evaluation_start = start + 1 if config.evaluation_protocol == "mstea" else start
    signed_errors: list[int] = []
    for name in train_names:
        dataset = Windows(
            data["states"], data["capacities"], data["cutoffs"],
            [name], config.seq_len, start=evaluation_start,
        )
        if len(dataset) == 0:
            continue
        _, _, _, rows = _evaluate(
            model, DataLoader(dataset, config.batch_size), config,
        )
        y_true = np.asarray([float(row["true_ah"]) for row in rows], dtype=np.float64)
        y_pred = np.asarray([float(row["pred_ah"]) for row in rows], dtype=np.float64)
        metrics = _capacity_metrics_for_protocol(y_true, y_pred, config)
        true_event = metrics.get("rul_true", metrics.get("rul_real", float("nan")))
        pred_event = metrics.get("rul_pred", float("nan"))
        if np.isfinite(true_event) and np.isfinite(pred_event):
            signed_errors.append(int(round(float(pred_event - true_event))))

    if not signed_errors:
        return 0, {"mode": mode, "num_phase_batteries": 0}
    delta = int(round(float(np.median(np.asarray(signed_errors, dtype=np.float64)))))
    if mode == "robust":
        clip = max(int(config.eol_event_phase_clip_cycles), 0)
        delta = int(np.clip(delta, -clip, clip))
    return delta, {
        "mode": mode,
        "num_phase_batteries": len(signed_errors),
        "train_signed_errors": ",".join(str(value) for value in signed_errors),
        "train_signed_error_median": float(np.median(signed_errors)),
    }


def _apply_eol_event_phase_alignment(
    metrics: dict[str, float],
    delta: int,
) -> dict[str, float]:
    """Adjust only the EOL event metrics; capacity predictions remain untouched."""
    true_key = "rul_true" if "rul_true" in metrics else "rul_real"
    true_event = float(metrics.get(true_key, float("nan")))
    pred_event = float(metrics.get("rul_pred", float("nan")))
    if not np.isfinite(true_event) or not np.isfinite(pred_event):
        return dict(metrics)
    patched = dict(metrics)
    patched["raw_ae"] = float(metrics["ae"])
    patched["raw_re"] = float(metrics["re"])
    patched["raw_rul_pred"] = pred_event
    adjusted_pred = max(pred_event - int(delta), 0.0)
    adjusted_ae = abs(adjusted_pred - true_event)
    patched["rul_pred"] = float(adjusted_pred)
    patched["ae"] = float(adjusted_ae)
    patched["re"] = float(min(adjusted_ae / max(true_event, 1.0), 1.0))
    return patched


def run_physics_mgi(data_dir: Path, output_dir: Path, config: PhysicsTrainConfig,
                    test_batteries: list[str] | None = None) -> dict[str, object]:
    torch.manual_seed(config.seed); np.random.seed(config.seed)
    if config.preprocessing_protocol not in {"legacy", "batter_moe"}:
        raise ValueError(
            "preprocessing_protocol must be one of: legacy, batter_moe"
        )
    summary_obj = np.load(
        _summary_path(data_dir, config.dataset, config.summary_filename), allow_pickle=True
    )[0]
    names = sorted(str(name) for name in summary_obj)
    folds, predictions, preprocessing_records = [], [], []
    for test in test_batteries or names:
        train_names = [n for n in names if n != test]
        data = load_states(data_dir, names, config)
        closure_solver = MicroPhysicalSolver(
            config.cutoff_voltage_v, config.discharge_current_a,
            config.tau_p_seconds, config.q_grid_max_ah, config.q_grid_points,
            ocp_profile=config.ocp_profile
        )
        for name in names:
            with torch.no_grad():
                reconstructed = closure_solver.capacity(
                    torch.from_numpy(data["states"][name]).float(),
                    data["cutoffs"][name],
                ).numpy()
            closure_mae = float(np.mean(np.abs(reconstructed - data["capacities"][name])))
            print(f"physics_closure battery={name} mae={closure_mae:.6f}Ah")
            if closure_mae > config.max_self_reconstruction_mae:
                raise RuntimeError(
                    f"Physical inversion closure failed for {name}: "
                    f"{closure_mae:.6f} Ah > {config.max_self_reconstruction_mae:.6f} Ah. "
                    "State-transition training was intentionally not started."
                )
        train = Windows(data["states"], data["capacities"], data["cutoffs"],
                        train_names, config.seq_len)
        val = Windows(data["states"], data["capacities"], data["cutoffs"],
                      train_names, config.seq_len,
                      tail=config.validation_fraction)
        val_keys = set(val.samples)
        train.samples = [s for s in train.samples if s not in val_keys]
        train_loader = DataLoader(train, config.batch_size, shuffle=True)
        val_loader = DataLoader(val, config.batch_size)
        use_minmax = config.state_scaling == "minmax" or (
            config.state_scaling == "protocol"
            and config.preprocessing_protocol == "batter_moe"
        )
        if use_minmax:
            # Fit Min-Max on the true training portion only. The validation
            # tail and held-out cell do not contribute to these statistics.
            train_parts = []
            fit_counts: dict[str, int] = {}
            for name in train_names:
                n_state = len(data["states"][name])
                split = int(n_state * (1.0 - config.validation_fraction))
                split = max(config.seq_len + 1, min(split, n_state))
                train_parts.append(data["states"][name][:split])
                fit_counts[name] = int(split)
            train_state_values = np.concatenate(train_parts, axis=0)
            state_offset_np, state_scale_np = fit_train_minmax(train_state_values)
            state_mean = torch.from_numpy(state_offset_np)
            state_std = torch.from_numpy(state_scale_np)
            scaling_method = "train-only min-max"
        else:
            train_state_values = np.concatenate(
                [data["states"][n] for n in train_names], axis=0
            )
            state_mean = torch.from_numpy(train_state_values.mean(axis=0))
            state_std = torch.from_numpy(train_state_values.std(axis=0).clip(1e-6))
            fit_counts = {name: int(len(data["states"][name])) for name in train_names}
            scaling_method = "legacy train-cell z-score"
        fold_preprocessing = {
            "test_battery": test,
            "protocol": config.preprocessing_protocol,
            "cleaned_summary": config.summary_filename,
            "capacity_scaling": (
                "C/C0"
                if config.capacity_target_scaling == "soh"
                or (
                    config.capacity_target_scaling == "protocol"
                    and config.preprocessing_protocol == "batter_moe"
                )
                else "absolute Ah"
            ),
            "rated_capacity_ah": float(config.rated_capacity),
            "state_feature_scaling": scaling_method,
            "state_offset": state_mean.tolist(),
            "state_scale": state_std.tolist(),
            "fit_cycle_counts": fit_counts,
            "validation_and_test_clipping": False,
        }
        preprocessing_records.append(fold_preprocessing)
        model = PhysicsGuidedStateModel(
            config.cutoff_voltage_v, config.discharge_current_a, config.tau_p_seconds,
            config.hidden_dim, config.num_layers, config.dropout, state_mean, state_std,
            config.q_grid_max_ah, config.q_grid_points, config.ocp_profile,
            config.thermo_step_scale, config.kinetic_step_scale,
            config.trend_short_window, config.trend_long_window
        )
        optim = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_eps,
        )
        if config.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim,
                T_max=max(config.epochs, 1),
                eta_min=config.lr * config.scheduler_min_lr_ratio,
            )
        elif config.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim,
                mode="min",
                factor=0.5,
                patience=config.scheduler_patience,
                min_lr=config.lr * config.scheduler_min_lr_ratio,
            )
        elif config.lr_scheduler == "none":
            scheduler = None
        else:
            raise ValueError(f"Unknown physics LR scheduler: {config.lr_scheduler}")
        best, best_state, best_epoch, stale = np.inf, None, 0, 0
        scale = torch.tensor([.25, .4, .4, .05, .05])
        for epoch in range(1, config.epochs + 1):
            model.train()
            losses, state_losses, curve_losses = [], [], []
            capacity_losses, direction_losses = [], []
            for batch in train_loader:
                out = model(
                    batch["states"], cutoff_voltage_v=batch["cutoff_voltage_v"]
                )
                state_loss = F.smooth_l1_loss((out["next_state"]-batch["next_state"])/scale,
                                              torch.zeros_like(batch["next_state"]))
                if config.state_supervision == "curve":
                    with torch.no_grad():
                        target_curve = model.solver.voltage_curve(
                            batch["next_state"]
                        )
                    curve_error = (
                        out["voltage_curve"] - target_curve
                    ) / config.voltage_error_scale
                    point_curve_loss = F.smooth_l1_loss(
                        curve_error, torch.zeros_like(curve_error),
                        beta=.2, reduction="none"
                    )
                    # No voltage observation exists beyond the measured cutoff.
                    # Excluding that region prevents supervision by an
                    # artificial post-cutoff OCP continuation.
                    observed = (
                        model.solver.q_grid.unsqueeze(0)
                        <= batch["capacity"].unsqueeze(1)
                    )
                    curve_loss = (
                        point_curve_loss * observed
                    ).sum() / observed.sum().clamp_min(1)
                else:
                    curve_loss = torch.zeros((), dtype=state_loss.dtype)
                late_weight = 1.0 + config.late_life_weight * (
                    1.0 - batch["capacity"] / config.rated_capacity
                ).clamp(0.0, 1.0)
                regeneration_weight = 1.0 + config.regeneration_loss_weight * (
                    (batch["capacity"] - batch["last_capacity"]) / 0.01
                ).clamp(0.0, 5.0)
                use_capacity_soh = config.capacity_target_scaling == "soh" or (
                    config.capacity_target_scaling == "protocol"
                    and config.preprocessing_protocol == "batter_moe"
                )
                if use_capacity_soh:
                    predicted_capacity = out["capacity_ah"] / config.rated_capacity
                    target_capacity = batch["capacity"] / config.rated_capacity
                    capacity_beta = config.capacity_huber_beta / config.rated_capacity
                else:
                    predicted_capacity = out["capacity_ah"]
                    target_capacity = batch["capacity"]
                    capacity_beta = config.capacity_huber_beta
                point_q_loss = F.smooth_l1_loss(
                    predicted_capacity, target_capacity,
                    beta=capacity_beta, reduction="none"
                )
                q_loss = (point_q_loss * late_weight * regeneration_weight).mean()
                last = batch["states"][:, -1]
                direction = F.relu(out["next_state"][:, :3]-last[:, :3]).mean()
                direction += F.relu(last[:, 3:]-out["next_state"][:, 3:]).mean()
                if config.state_supervision == "curve":
                    loss = config.curve_loss_weight * curve_loss
                    loss += config.weak_state_loss_weight * state_loss
                    loss += config.capacity_loss_weight * q_loss
                else:
                    loss = config.state_loss_weight*state_loss + config.capacity_loss_weight*q_loss
                loss += config.direction_loss_weight*direction
                optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optim.step()
                losses.append(float(loss.detach()))
                state_losses.append(float(state_loss.detach()))
                curve_losses.append(float(curve_loss.detach()))
                capacity_losses.append(float(q_loss.detach()))
                direction_losses.append(float(direction.detach()))
            mae, rmse, r2, val_rows = _evaluate(model, val_loader, config)
            val_persistence = float(np.mean([
                abs(r["persistence_ah"] - r["true_ah"]) for r in val_rows
            ]))
            print(f"battery={test} epoch={epoch}/{config.epochs} "
                  f"loss={np.mean(losses):.6f} "
                  f"state={np.mean(state_losses):.6f} "
                  f"q={np.mean(capacity_losses):.6f} "
                  f"curve={np.mean(curve_losses):.6f} "
                  f"direction={np.mean(direction_losses):.6f} "
                  f"val_mae={mae:.6f}Ah val_rmse={rmse:.6f}Ah val_r2={r2:.4f} "
                  f"val_persist={val_persistence:.6f}Ah")
            if config.lr_scheduler == "plateau":
                scheduler.step(mae)
            elif scheduler is not None:
                scheduler.step()
            if mae < best - 1e-6:
                best, best_epoch, stale = mae, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= config.early_stopping_patience:
                    print(f"battery={test} early_stop epoch={epoch} "
                          f"best_val_mae={best:.6f}Ah")
                    break
        if best_state: model.load_state_dict(best_state)
        epa_delta, epa_info = _estimate_eol_event_phase_delta(
            model, data, train_names, config
        )
        for start in config.start_points:
            # Under the MSTE-A/BATTER-MoE convention, SP=k means that cycle k
            # is the last observed point and the first target is cycle k+1.
            evaluation_start = start + 1 if config.evaluation_protocol == "mstea" else start
            ds = Windows(data["states"], data["capacities"], data["cutoffs"],
                         [test], config.seq_len, start=evaluation_start)
            mae, rmse, r2, rows = _evaluate(
                model, DataLoader(ds, config.batch_size), config
            )
            persistence = float(np.mean([abs(r["persistence_ah"]-r["true_ah"]) for r in rows]))
            y_true = np.asarray([r["true_ah"] for r in rows], dtype=np.float64)
            y_pred = np.asarray([r["pred_ah"] for r in rows], dtype=np.float64)
            patch = _capacity_metrics_for_protocol(
                y_true, y_pred, config
            )
            if epa_delta != 0:
                patch = _apply_eol_event_phase_alignment(patch, epa_delta)
            folds.append({"test_battery": test, "start_point": start, "num_windows": len(rows),
                          "mae": mae, "rmse": rmse, "r2": r2,
                          "RUL_true": patch.get("rul_true", patch.get("rul_real", float("nan"))),
                          "RUL_pred": patch.get("rul_pred", float("nan")),
                          "AE": patch["ae"], "RE": patch["re"],
                          "raw_AE": patch.get("raw_ae", patch["ae"]),
                          "raw_RE": patch.get("raw_re", patch["re"]),
                          "eol_phase_delta": int(epa_delta),
                          "eol_phase_mode": epa_info.get("mode", "none"),
                          "eol_phase_train_signed_errors": epa_info.get("train_signed_errors", ""),
                          "best_validation_mae": float(best),
                          "best_epoch": int(best_epoch),
                          "persistence_mae": persistence})
            predictions += [
                {**r, "start_point": start, "eol_phase_delta": int(epa_delta)}
                for r in rows
            ]
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "config": asdict(config),
                    "state_names": STATE_NAMES, "preprocessing": fold_preprocessing},
                   output_dir / f"{test}_physics.pt")
    payload = {"model": "physical_inverse_neural_transition_physical_forward",
               "state_names": STATE_NAMES, "config": asdict(config),
               "preprocessing": preprocessing_records, "folds": folds}
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "preprocessing.json").write_text(
        json.dumps(preprocessing_records, indent=2), encoding="utf-8"
    )
    if predictions:
        with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(predictions[0])); writer.writeheader()
            writer.writerows(predictions)
    return payload


def format_physics_results(payload: dict[str, object]) -> str:
    lines = ["battery      start   windows       MAE       RMSE       R2      AE        RE    persistence"]
    for r in payload["folds"]:
        lines.append(f"{r['test_battery']:<12}{r['start_point']:>6}{r['num_windows']:>10}"
                     f"{r['mae']:>10.6f}{r['rmse']:>11.6f}{r['r2']:>9.4f}"
                     f"{r['AE']:>8}{r['RE']:>10.4f}{r['persistence_mae']:>15.6f}")
    return "\n".join(lines)
