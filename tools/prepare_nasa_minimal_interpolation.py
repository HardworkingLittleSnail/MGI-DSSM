from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mgi_dssm.raw_nasa import NASA_BATTERIES, prepare_nasa_raw_dataset


# One-based cycle numbers. Each point is an isolated upward measurement spike
# and is replaced by the linear interpolation of its two immediate neighbours.
REPAIR_CYCLES = {
    "B0005": (31, 90, 151),
    "B0006": (90,),
    "B0007": (90, 151),
    "B0018": (121,),
}
RATED_CAPACITY_AH = 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_summary(path: Path, summary: dict[str, pd.DataFrame]) -> None:
    payload = np.empty(1, dtype=object)
    payload[0] = summary
    np.save(path, payload, allow_pickle=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the minimally repaired NASA capacity summary from the official "
            "MAT files without deleting or renumbering any cycle."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-name", default="NASA_Data_minimal_interpolated.npy"
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    raw_dir = data_dir / "raw" / "NASA data"
    processed_dir = data_dir / "processed" / "NASA data"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # force=True guarantees that the summary is rebuilt from the official MAT
    # files rather than loaded from a previously generated summary.
    raw_summary, _ = prepare_nasa_raw_dataset(data_dir, force=True)
    cleaned: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, object]] = []

    for battery in NASA_BATTERIES:
        frame = raw_summary[battery].copy().sort_values("Cycle").reset_index(drop=True)
        raw_capacity = pd.to_numeric(frame["Capacity"], errors="raise").to_numpy(
            dtype=np.float64
        )
        capacity = raw_capacity.copy()
        for cycle in REPAIR_CYCLES[battery]:
            index = cycle - 1
            if index <= 0 or index >= len(capacity) - 1:
                raise ValueError(f"Cannot interpolate endpoint: {battery} cycle={cycle}")
            if int(frame.iloc[index]["Cycle"]) != cycle:
                raise ValueError(f"Cycle alignment mismatch: {battery} cycle={cycle}")
            repaired = float((raw_capacity[index - 1] + raw_capacity[index + 1]) / 2.0)
            audit_rows.append(
                {
                    "battery": battery,
                    "cycle": cycle,
                    "row_index": index,
                    "previous_capacity_ah": float(raw_capacity[index - 1]),
                    "raw_capacity_ah": float(raw_capacity[index]),
                    "next_capacity_ah": float(raw_capacity[index + 1]),
                    "cleaned_capacity_ah": repaired,
                    "action": "linear_interpolation_of_immediate_neighbours",
                }
            )
            capacity[index] = repaired
        frame["Capacity"] = capacity
        frame["Capacity_SOH"] = capacity / RATED_CAPACITY_AH
        if len(frame) != len(raw_summary[battery]):
            raise AssertionError(f"Cycle count changed unexpectedly: {battery}")
        cleaned[battery] = frame

    expected_repairs = sum(len(cycles) for cycles in REPAIR_CYCLES.values())
    if len(audit_rows) != expected_repairs:
        raise AssertionError(
            f"Expected {expected_repairs} repaired points, got {len(audit_rows)}"
        )

    output_path = processed_dir / args.output_name
    audit_path = processed_dir / "nasa_minimal_interpolation_audit.csv"
    metadata_path = processed_dir / "nasa_minimal_interpolation.json"
    save_summary(output_path, cleaned)

    with audit_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    total_cycles = sum(len(frame) for frame in cleaned.values())
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_policy": "official NASA MAT files only",
        "source_directory": str(raw_dir),
        "source_sha256": {
            f"{battery}.mat": sha256(raw_dir / f"{battery}.mat")
            for battery in NASA_BATTERIES
        },
        "output": str(output_path),
        "method": "minimal isolated-spike repair by immediate-neighbour linear interpolation",
        "rated_capacity_ah": RATED_CAPACITY_AH,
        "deleted_cycles": 0,
        "renumbered_cycles": 0,
        "total_cycles": total_cycles,
        "repaired_points": expected_repairs,
        "repaired_fraction": expected_repairs / total_cycles,
        "repair_cycles_one_based": {
            battery: list(cycles) for battery, cycles in REPAIR_CYCLES.items()
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved: {output_path}")
    print(f"audit: {audit_path}")
    print(
        f"cycles={total_cycles}, repaired={expected_repairs}, "
        f"deleted=0, repaired_fraction={expected_repairs / total_cycles:.6%}"
    )


if __name__ == "__main__":
    main()
