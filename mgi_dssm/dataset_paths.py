from __future__ import annotations

from pathlib import Path


DATASET_DIRECTORIES = {
    "calce": "CALCE data",
    "nasa": "NASA data",
    "tju": "TJU data",
}


def _name(dataset: str) -> str:
    try:
        return DATASET_DIRECTORIES[dataset.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset}") from exc


def raw_dataset_dir(data_dir: Path, dataset: str) -> Path:
    """Resolve the immutable official-raw directory, with legacy fallback."""
    base = Path(data_dir)
    if base.name.lower() == "processed":
        modern = base.parent / "raw" / _name(dataset)
    else:
        modern = base / "raw" / _name(dataset)
    if modern.exists():
        return modern
    return base / _name(dataset)


def processed_dataset_dir(data_dir: Path, dataset: str) -> Path:
    """Resolve the generated-data directory, with legacy fallback."""
    base = Path(data_dir)
    if base.name.lower() == "processed":
        return base / _name(dataset)
    modern_root = base / "processed"
    if modern_root.exists() or (base / "raw").exists():
        return modern_root / _name(dataset)
    return base / _name(dataset)
