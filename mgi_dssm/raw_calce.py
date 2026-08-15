from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Iterable
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

from .dataset_paths import processed_dataset_dir, raw_dataset_dir


NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

CALCE_BATTERIES = ("CS2_35", "CS2_36", "CS2_37", "CS2_38")
CALCE_RAW_SUMMARY_NAME = "CALCE_Data_raw_unfiltered.npy"
CALCE_RAW_CURVE_CACHE_NAME = "raw_discharge_curves_batter_moe_v1.npy"


def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        data = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(data)
    values = []
    for item in root.findall(NS + "si"):
        values.append("".join((t.text or "") for t in item.findall(".//" + NS + "t")))
    return values


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    value = cell.find(NS + "v")
    if value is None:
        return ""
    text = value.text or ""
    if cell.attrib.get("t") == "s" and text.isdigit():
        idx = int(text)
        if idx < len(shared):
            return shared[idx]
    return text


def _column_index(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    out = 0
    for ch in letters:
        out = out * 26 + ord(ch.upper()) - 64
    return out - 1


def iter_xlsx_rows(path: Path) -> Iterable[dict[str, str]]:
    """Yield rows from the largest worksheet using only the standard library."""

    with zipfile.ZipFile(path) as zf:
        shared = _shared_strings(zf)
        sheets = [
            name
            for name in zf.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        ]
        if not sheets:
            return
        sheet = max(sheets, key=lambda name: zf.getinfo(name).file_size)
        root = ET.fromstring(zf.read(sheet))
        rows = root.findall(NS + "sheetData/" + NS + "row")
        if not rows:
            return

        header: list[str] = []
        for cell in rows[0].findall(NS + "c"):
            idx = _column_index(cell.attrib.get("r", "A1"))
            while len(header) <= idx:
                header.append("")
            header[idx] = _cell_value(cell, shared)

        for row in rows[1:]:
            values = [""] * len(header)
            for cell in row.findall(NS + "c"):
                idx = _column_index(cell.attrib.get("r", "A1"))
                if idx < len(values):
                    values[idx] = _cell_value(cell, shared)
            yield dict(zip(header, values))


def _number(row: dict[str, str], key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _first_timestamp(path: Path) -> float:
    try:
        first = next(iter(iter_xlsx_rows(path)))
        return float(first.get("Date_Time", ""))
    except (StopIteration, TypeError, ValueError):
        parts = path.stem.rsplit("_", 3)
        try:
            return float(date(2000 + int(parts[-1]), int(parts[-3]), int(parts[-2])).toordinal())
        except (ValueError, IndexError):
            return 0.0


def prepare_calce_raw_dataset(
    data_dir: Path, force: bool = False
) -> tuple[dict[str, pd.DataFrame], dict[str, list[dict[str, object]]]]:
    """Extract every measurable CALCE discharge cycle before outlier cleaning."""

    calce_dir = raw_dataset_dir(data_dir, "calce")
    output_dir = processed_dataset_dir(data_dir, "calce")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / CALCE_RAW_SUMMARY_NAME
    curves_path = output_dir / CALCE_RAW_CURVE_CACHE_NAME
    if not force and summary_path.exists() and curves_path.exists():
        payload = np.load(summary_path, allow_pickle=True)
        summary = payload.item() if payload.shape == () else payload[0]
        curves = np.load(curves_path, allow_pickle=True).item()
        if sorted(summary) == sorted(curves) == sorted(CALCE_BATTERIES):
            return summary, curves

    summaries: dict[str, pd.DataFrame] = {}
    all_curves: dict[str, list[dict[str, object]]] = {}
    for battery in CALCE_BATTERIES:
        curves: list[dict[str, object]] = []
        source_files: list[str] = []
        raw_cycles: list[int] = []
        paths = sorted((calce_dir / battery).glob("*.xlsx"), key=_first_timestamp)
        for path in paths:
            groups: dict[int, list[dict[str, str]]] = {}
            for row in iter_xlsx_rows(path):
                try:
                    groups.setdefault(int(float(row["Cycle_Index"])), []).append(row)
                except (KeyError, ValueError):
                    continue
            for cycle in sorted(groups):
                rows = [
                    row
                    for row in groups[cycle]
                    if int(_number(row, "Step_Index", -1)) == 7
                ]
                if len(rows) < 2:
                    continue
                time = np.asarray([_number(row, "Test_Time(s)") for row in rows])
                voltage = np.asarray([_number(row, "Voltage(V)") for row in rows])
                signed_current = np.asarray([_number(row, "Current(A)") for row in rows])
                resistance = np.asarray(
                    [_number(row, "Internal_Resistance(Ohm)") for row in rows]
                )
                valid = np.isfinite(time) & np.isfinite(voltage) & np.isfinite(signed_current)
                time, voltage, signed_current, resistance = (
                    time[valid], voltage[valid], signed_current[valid], resistance[valid]
                )
                if len(time) < 2:
                    continue
                dt = np.diff(time)
                dq = np.where((dt > 0.0) & (dt <= 60.0), -dt * signed_current[1:] / 3600.0, 0.0)
                q = np.cumsum(dq)
                voltage = voltage[1:]
                current = np.abs(signed_current[1:])
                resistance = resistance[1:]
                valid_q = np.isfinite(q) & np.isfinite(voltage) & np.isfinite(current) & (q >= 0.0)
                q, voltage, current, resistance = (
                    q[valid_q], voltage[valid_q], current[valid_q], resistance[valid_q]
                )
                if len(q) < 2 or q[-1] <= 0.0:
                    continue
                observed_r = resistance[np.isfinite(resistance) & (resistance > 1e-6)]
                curves.append(
                    {
                        "q": q.astype(np.float64),
                        "v": voltage.astype(np.float64),
                        "current": float(np.median(current)),
                        "r0": float(np.median(observed_r)) if observed_r.size else 0.05,
                    }
                )
                source_files.append(path.name)
                raw_cycles.append(int(cycle))
        capacities = np.asarray([float(curve["q"][-1]) for curve in curves], dtype=np.float64)
        summaries[battery] = pd.DataFrame(
            {
                "BatteryName": battery,
                "Cycle": np.arange(1, len(curves) + 1, dtype=np.int64),
                "source_file": source_files,
                "raw_cycle_index": raw_cycles,
                "Capacity": capacities,
            }
        )
        all_curves[battery] = curves
        print(
            f"calce_raw battery={battery} cycles={len(curves)} "
            f"capacity_range={capacities.min():.6f}-{capacities.max():.6f}Ah"
        )

    payload = np.empty(1, dtype=object)
    payload[0] = summaries
    np.save(summary_path, payload, allow_pickle=True)
    np.save(curves_path, all_curves, allow_pickle=True)
    return summaries, all_curves


def audit_raw_files(data_dir: Path, max_files: int = 4) -> Dict[str, object]:
    paths = sorted(raw_dataset_dir(data_dir, "calce").glob("CS2_*/*.xlsx"))[:max_files]
    totals: dict[str, int] = defaultdict(int)
    step_counts: Counter[int] = Counter()
    fields = [
        "Current(A)",
        "Voltage(V)",
        "Internal_Resistance(Ohm)",
        "AC_Impedance(Ohm)",
        "ACI_Phase_Angle(Deg)",
        "Discharge_Capacity(Ah)",
        "Charge_Capacity(Ah)",
    ]

    for path in paths:
        totals["files"] += 1
        for row in iter_xlsx_rows(path):
            totals["rows"] += 1
            try:
                step_counts[int(float(row.get("Step_Index", "")))] += 1
            except ValueError:
                pass
            for field in fields:
                value = row.get(field, "")
                if value != "":
                    totals[field + "_nonempty"] += 1
                    try:
                        if abs(float(value)) > 1e-12:
                            totals[field + "_nonzero"] += 1
                    except ValueError:
                        pass

    return {
        "files": totals["files"],
        "rows": totals["rows"],
        "step_counts": dict(sorted(step_counts.items())),
        "field_counts": {field: {"nonempty": totals[field + "_nonempty"], "nonzero": totals[field + "_nonzero"]} for field in fields},
    }
