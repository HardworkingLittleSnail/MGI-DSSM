from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (ROOT / "data").resolve()
STAGE_ROOT = DATA_ROOT / ".official_dataset_stage"
DOWNLOAD_ROOT = STAGE_ROOT / "downloads"
REPLACEMENT_ROOT = STAGE_ROOT / "replacement"
BACKUP_ROOT = STAGE_ROOT / "backup"

SOURCES = {
    "calce": {
        "page": "https://web.calce.umd.edu/batteries/data/",
        "archives": [
            ("CS2_35.zip", "https://web.calce.umd.edu/batteries/data/CS2_35.zip", None),
            ("CS2_36.zip", "https://web.calce.umd.edu/batteries/data/CS2_36.zip", None),
            ("CS2_37.zip", "https://web.calce.umd.edu/batteries/data/CS2_37.zip", None),
            ("CS2_38.zip", "https://web.calce.umd.edu/batteries/data/CS2_38.zip", None),
        ],
    },
    "nasa": {
        "page": "https://data.nasa.gov/dataset/li-ion-battery-aging-datasets",
        "archives": [
            (
                "NASA_Battery_Data_Set.zip",
                "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip",
                None,
            )
        ],
    },
    "tju": {
        "page": "https://zenodo.org/records/6405084",
        "archives": [
            (
                "Dataset_3_NCM_NCA_battery.zip",
                "https://zenodo.org/records/6405084/files/Dataset_3_NCM_NCA_battery.zip?download=1",
                "d11e68e410a638058906af5e2f5f60f3",
            )
        ],
    },
}

NASA_FILES = ("B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat")
TJU_FILES = ("CY25-05_1-#1.csv", "CY25-05_1-#2.csv", "CY25-05_1-#3.csv")


def digest(path: Path, algorithm: str = "sha256") -> str:
    checksum = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def download(url: str, destination: Path, attempts: int = 4) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Use downloaded archive: {destination} ({destination.stat().st_size:,} bytes)")
        return
    for attempt in range(1, attempts + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "Mozilla/5.0 (official-dataset-retriever/1.0)"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", 200)
                if offset and status != 206:
                    offset = 0
                    part.unlink(missing_ok=True)
                mode = "ab" if offset else "wb"
                expected = response.headers.get("Content-Length")
                total = offset + int(expected) if expected else None
                copied = offset
                next_report = copied + 32 * 1024 * 1024
                with part.open(mode) as output:
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        copied += len(block)
                        if copied >= next_report:
                            suffix = f"/{total:,}" if total else ""
                            print(f"  {destination.name}: {copied:,}{suffix} bytes", flush=True)
                            next_report += 32 * 1024 * 1024
                if total is not None and copied != total:
                    raise IOError(f"incomplete response: expected {total}, got {copied}")
            os.replace(part, destination)
            print(f"Downloaded: {destination} ({destination.stat().st_size:,} bytes)")
            return
        except (OSError, urllib.error.URLError) as exc:
            print(f"Download attempt {attempt}/{attempts} failed for {url}: {exc}", file=sys.stderr)
            if attempt == attempts:
                raise
            time.sleep(min(5 * attempt, 15))


def safe_member_name(name: str) -> bool:
    path = Path(name.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts


def find_members(archive: zipfile.ZipFile, basename: str) -> list[zipfile.ZipInfo]:
    return [
        item
        for item in archive.infolist()
        if not item.is_dir()
        and safe_member_name(item.filename)
        and Path(item.filename.replace("\\", "/")).name.lower() == basename.lower()
    ]


def copy_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, target.open("wb") as output:
        shutil.copyfileobj(source, output, length=4 * 1024 * 1024)


def extract_calce(archives: list[Path], target: Path) -> dict[str, object]:
    counts: dict[str, int] = {}
    for archive_path in archives:
        battery = archive_path.stem
        battery_dir = target / battery
        battery_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            bad = archive.testzip()
            if bad:
                raise zipfile.BadZipFile(f"CRC failure in {archive_path}: {bad}")
            members = [
                item
                for item in archive.infolist()
                if not item.is_dir()
                and safe_member_name(item.filename)
                and Path(item.filename.replace("\\", "/")).suffix.lower() == ".xlsx"
            ]
            if not members:
                raise ValueError(f"No XLSX files found in {archive_path}")
            seen: set[str] = set()
            for member in members:
                name = Path(member.filename.replace("\\", "/")).name
                if name.lower() in seen:
                    raise ValueError(f"Duplicate CALCE basename in {archive_path}: {name}")
                seen.add(name.lower())
                copy_member(archive, member, battery_dir / name)
            counts[battery] = len(members)
    return {"raw_files": counts}


def extract_named_files(archive_path: Path, names: tuple[str, ...], target: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise zipfile.BadZipFile(f"CRC failure in {archive_path}: {bad}")
        for name in names:
            matches = find_members(archive, name)
            if len(matches) == 1:
                copy_member(archive, matches[0], target / name)
            elif not matches:
                nested_matches: list[bytes] = []
                for nested in archive.infolist():
                    if nested.is_dir() or not nested.filename.lower().endswith(".zip"):
                        continue
                    with archive.open(nested) as stream:
                        nested_payload = stream.read()
                    with zipfile.ZipFile(io.BytesIO(nested_payload)) as nested_archive:
                        inner = find_members(nested_archive, name)
                        for member in inner:
                            nested_matches.append(nested_archive.read(member))
                if len(nested_matches) != 1:
                    raise ValueError(
                        f"Expected exactly one {name} in {archive_path} or its nested ZIPs, "
                        f"found {len(nested_matches)}"
                    )
                (target / name).parent.mkdir(parents=True, exist_ok=True)
                (target / name).write_bytes(nested_matches[0])
            else:
                raise ValueError(f"Expected exactly one {name} in {archive_path}, found {len(matches)}")
            sizes[name] = (target / name).stat().st_size
    return sizes


def validate_replacement(root: Path) -> dict[str, object]:
    calce = root / "CALCE data"
    nasa = root / "NASA data"
    tju = root / "TJU data"
    calce_counts = {
        battery: len(list((calce / battery).glob("*.xlsx")))
        for battery in ("CS2_35", "CS2_36", "CS2_37", "CS2_38")
    }
    if any(count <= 0 for count in calce_counts.values()):
        raise ValueError(f"Incomplete CALCE extraction: {calce_counts}")
    nasa_sizes = {name: (nasa / name).stat().st_size for name in NASA_FILES}
    if any(size < 1_000_000 for size in nasa_sizes.values()):
        raise ValueError(f"Implausible NASA MAT sizes: {nasa_sizes}")
    tju_sizes = {name: (tju / name).stat().st_size for name in TJU_FILES}
    if any(size < 10_000_000 for size in tju_sizes.values()):
        raise ValueError(f"Implausible TJU CSV sizes: {tju_sizes}")
    expected_header = {"time/s", "Ecell/V", "<I>/mA", "cycle number"}
    for name in TJU_FILES:
        with (tju / name).open("r", encoding="utf-8-sig", errors="replace") as stream:
            header = set(stream.readline().strip().split(","))
        if not expected_header.issubset(header):
            raise ValueError(f"Unexpected TJU header in {name}: {sorted(header)}")
    return {"calce_xlsx_counts": calce_counts, "nasa_mat_sizes": nasa_sizes, "tju_csv_sizes": tju_sizes}


def build_replacement() -> dict[str, object]:
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    if REPLACEMENT_ROOT.exists():
        shutil.rmtree(REPLACEMENT_ROOT)
    REPLACEMENT_ROOT.mkdir(parents=True)
    archive_records: list[dict[str, object]] = []
    downloaded: dict[str, list[Path]] = {}
    for dataset, source in SOURCES.items():
        downloaded[dataset] = []
        for filename, url, expected_md5 in source["archives"]:
            path = DOWNLOAD_ROOT / filename
            download(url, path)
            actual_md5 = digest(path, "md5")
            if expected_md5 and actual_md5.lower() != expected_md5.lower():
                raise ValueError(f"MD5 mismatch for {path}: expected {expected_md5}, got {actual_md5}")
            record = {
                "dataset": dataset,
                "filename": filename,
                "url": url,
                "source_page": source["page"],
                "bytes": path.stat().st_size,
                "md5": actual_md5,
                "sha256": digest(path, "sha256"),
            }
            archive_records.append(record)
            downloaded[dataset].append(path)
            print(f"Verified archive: {filename} sha256={record['sha256']}")

    extraction: dict[str, object] = {}
    extraction["calce"] = extract_calce(downloaded["calce"], REPLACEMENT_ROOT / "CALCE data")
    extraction["nasa"] = extract_named_files(
        downloaded["nasa"][0], NASA_FILES, REPLACEMENT_ROOT / "NASA data"
    )
    extraction["tju"] = extract_named_files(
        downloaded["tju"][0], TJU_FILES, REPLACEMENT_ROOT / "TJU data"
    )
    validation = validate_replacement(REPLACEMENT_ROOT)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "official raw archives; no third-party preprocessed summaries",
        "archives": archive_records,
        "extraction": extraction,
        "validation": validation,
    }
    for directory in ("CALCE data", "NASA data", "TJU data"):
        (REPLACEMENT_ROOT / directory / "official_source_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return manifest


def replace_directories() -> None:
    raw_root = DATA_ROOT / "raw"
    processed_root = DATA_ROOT / "processed"
    names = ("CALCE data", "NASA data", "TJU data")
    targets = [raw_root / name for name in names]
    sources = [REPLACEMENT_ROOT / name for name in names]
    for target, source in zip(targets, sources):
        if target.resolve().parent != raw_root.resolve() or source.resolve().parent != REPLACEMENT_ROOT.resolve():
            raise RuntimeError(f"Unsafe replacement boundary: {target} <- {source}")
        if not source.is_dir():
            raise FileNotFoundError(f"Validated replacement missing: {source}")

    validate_replacement(REPLACEMENT_ROOT)
    if BACKUP_ROOT.exists():
        shutil.rmtree(BACKUP_ROOT)
    BACKUP_ROOT.mkdir(parents=True)
    moved_old: list[tuple[Path, Path]] = []
    moved_new: list[tuple[Path, Path]] = []
    try:
        old_locations = (
            [(DATA_ROOT / name, BACKUP_ROOT / "legacy" / name) for name in names]
            + [(raw_root / name, BACKUP_ROOT / "raw" / name) for name in names]
            + [(processed_root / name, BACKUP_ROOT / "processed" / name) for name in names]
        )
        for old, backup in old_locations:
            if old.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                old.rename(backup)
                moved_old.append((backup, old))
        raw_root.mkdir(parents=True, exist_ok=True)
        processed_root.mkdir(parents=True, exist_ok=True)
        for source, target in zip(sources, targets):
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            moved_new.append((target, source))
        validate_replacement(raw_root)
        for name in names:
            (processed_root / name).mkdir(parents=True, exist_ok=True)
    except Exception:
        for target, source in reversed(moved_new):
            if target.exists() and not source.exists():
                target.rename(source)
        for backup, target in reversed(moved_old):
            if backup.exists() and not target.exists():
                backup.rename(target)
        raise
    shutil.rmtree(BACKUP_ROOT)
    print("Replaced legacy dataset directories.")
    print(f"Official raw data: {raw_root}")
    print(f"Generated preprocessing outputs: {processed_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, validate, and replace all three dataset directories.")
    parser.add_argument("--download-only", action="store_true", help="Prepare and validate staging data without replacing directories.")
    args = parser.parse_args()
    manifest = build_replacement()
    print(json.dumps(manifest["validation"], indent=2, ensure_ascii=False))
    if not args.download_only:
        replace_directories()


if __name__ == "__main__":
    main()
