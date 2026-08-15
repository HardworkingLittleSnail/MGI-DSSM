"""Build cycle-aligned IC2ML charge-curve features from the raw CALCE files."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np


NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [
        "".join((node.text or "") for node in item.findall(".//" + NS + "t"))
        for item in root.findall(NS + "si")
    ]


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    value = cell.find(NS + "v")
    if value is None:
        return ""
    text = value.text or ""
    if cell.attrib.get("t") == "s" and text.isdigit():
        return shared[int(text)]
    return text


def _column_index(reference: str) -> int:
    result = 0
    for character in "".join(ch for ch in reference if ch.isalpha()):
        result = result * 26 + ord(character.upper()) - 64
    return result - 1


def iter_xlsx_rows(path: Path):
    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive)
        sheets = [
            name
            for name in archive.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        ]
        sheet = max(sheets, key=lambda name: archive.getinfo(name).file_size)
        rows = ET.fromstring(archive.read(sheet)).findall(NS + "sheetData/" + NS + "row")
        if not rows:
            return
        header: list[str] = []
        for cell in rows[0].findall(NS + "c"):
            index = _column_index(cell.attrib.get("r", "A1"))
            while len(header) <= index:
                header.append("")
            header[index] = _cell_value(cell, shared)
        for row in rows[1:]:
            values = [""] * len(header)
            for cell in row.findall(NS + "c"):
                index = _column_index(cell.attrib.get("r", "A1"))
                if index < len(values):
                    values[index] = _cell_value(cell, shared)
            yield dict(zip(header, values))


def _number(row: dict[str, str], name: str, default: float = np.nan) -> float:
    try:
        return float(row.get(name, ""))
    except (TypeError, ValueError):
        return default


def _first_timestamp(path: Path) -> float:
    try:
        return _number(next(iter(iter_xlsx_rows(path))), "Date_Time", 0.0)
    except StopIteration:
        return 0.0


def _charge_increment(
    rows: list[dict[str, str]], voltage_start: float, voltage_end: float
) -> np.ndarray | None:
    voltage = np.asarray([_number(row, "Voltage(V)") for row in rows])
    current = np.asarray([_number(row, "Current(A)") for row in rows])
    capacity = np.asarray([_number(row, "Charge_Capacity(Ah)") for row in rows])
    valid = np.isfinite(voltage) & np.isfinite(current) & np.isfinite(capacity) & (current > 0)
    voltage, capacity = voltage[valid], capacity[valid]
    if len(voltage) < 2 or voltage.min() > voltage_start or voltage.max() < voltage_end:
        return None
    order = np.argsort(voltage)
    voltage, capacity = voltage[order], capacity[order]
    voltage, unique = np.unique(voltage, return_index=True)
    capacity = capacity[unique]
    sampled = np.interp(np.linspace(voltage_start, voltage_end, 10), voltage, capacity)
    return (sampled - sampled[0]).astype(np.float32)


def _discharge_capacity(rows: list[dict[str, str]]) -> float | None:
    rows = [row for row in rows if int(_number(row, "Step_Index", -1)) == 7]
    if len(rows) < 2:
        return None
    time = np.asarray([_number(row, "Test_Time(s)") for row in rows])
    current = np.asarray([_number(row, "Current(A)") for row in rows])
    valid = np.isfinite(time) & np.isfinite(current)
    time, current = time[valid], current[valid]
    if len(time) < 2:
        return None
    increments = -np.diff(time) * current[1:] / 3600.0
    increments = increments[np.isfinite(increments)]
    if len(increments) == 0:
        return None
    capacity = float(np.sum(increments))
    return capacity if capacity > 0 else None


def _align(raw: list[dict[str, object]], target: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray]:
    source = np.asarray([row["discharge_capacity"] for row in raw], dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n, m = len(source), len(target)
    if n < m:
        raise ValueError(f"cannot align {m} summary cycles to only {n} raw cycles")
    previous = np.full(m + 1, np.inf)
    previous[0] = 0.0
    take = np.zeros((n + 1, m + 1), dtype=np.bool_)
    for i in range(1, n + 1):
        current = np.full(m + 1, np.inf)
        current[0] = 0.0
        for j in range(1, min(i, m) + 1):
            skip = previous[j]
            match = previous[j - 1] + (source[i - 1] - target[j - 1]) ** 2
            if match <= skip:
                current[j], take[i, j] = match, True
            else:
                current[j] = skip
        previous = current
    indices: list[int] = []
    i, j = n, m
    while j:
        if take[i, j]:
            indices.append(i - 1)
            i, j = i - 1, j - 1
        else:
            i -= 1
    indices.reverse()
    return [raw[index] for index in indices], np.abs(source[indices] - target)


def build(root: Path, voltage_start: float, voltage_end: float) -> Path:
    data_dir = root / "data" / "CALCE data"
    summary = np.load(data_dir / "CALCE_Data.npy", allow_pickle=True)[0]
    output: dict[str, dict[str, np.ndarray]] = {}
    for battery, frame in summary.items():
        raw: list[dict[str, object]] = []
        paths = sorted((data_dir / str(battery)).glob("*.xlsx"), key=_first_timestamp)
        for path in paths:
            groups: dict[int, list[dict[str, str]]] = {}
            for row in iter_xlsx_rows(path):
                cycle = _number(row, "Cycle_Index")
                if np.isfinite(cycle):
                    groups.setdefault(int(cycle), []).append(row)
            for cycle in sorted(groups):
                discharge_capacity = _discharge_capacity(groups[cycle])
                if discharge_capacity is not None:
                    raw.append(
                        {
                            "discharge_capacity": discharge_capacity,
                            "increment": _charge_increment(
                                groups[cycle], voltage_start, voltage_end
                            ),
                        }
                    )
        capacities = frame["Capacity"].to_numpy(dtype=np.float32)
        aligned, errors = _align(raw, capacities)
        increments = [row["increment"] for row in aligned]
        imputed = np.asarray([value is None for value in increments], dtype=np.bool_)
        missing = sum(value is None for value in increments)
        last_valid = None
        for index, value in enumerate(increments):
            if value is not None:
                last_valid = value
            elif last_valid is not None:
                increments[index] = last_valid.copy()
        if all(value is None for value in increments):
            raise ValueError(f"{battery} has no charge curve covering the selected interval")
        # Never use a later cycle to fill an early input.  Zero represents no
        # observed capacity increment and is the only causal leading fallback.
        for index, value in enumerate(increments):
            if value is None:
                increments[index] = np.zeros(10, dtype=np.float32)
            else:
                break
        output[str(battery)] = {
            "increments": np.stack(increments).astype(np.float32),
            "capacities": capacities,
            "cycles": frame["Cycle"].to_numpy(dtype=np.int64),
            "imputed": imputed,
        }
        print(
            f"{battery}: raw={len(raw)} aligned={len(aligned)} missing_charge={missing} "
            f"align_mae={errors.mean():.8f}Ah align_max={errors.max():.8f}Ah",
            flush=True,
        )
    output_path = data_dir / f"CALCE_IC2ML_charge_{voltage_start:g}-{voltage_end:g}.npy"
    payload = np.empty(1, dtype=object)
    payload[0] = output
    np.save(output_path, payload, allow_pickle=True)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voltage-start", type=float, default=3.6)
    parser.add_argument("--voltage-end", type=float, default=3.7)
    arguments = parser.parse_args()
    if not np.isclose(arguments.voltage_end - arguments.voltage_start, 0.1):
        parser.error("IC2ML requires a 0.1 V sampling interval")
    print(build(Path(__file__).resolve().parent, arguments.voltage_start, arguments.voltage_end))
