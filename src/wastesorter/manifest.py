"""Baca/tulis manifest split (data/manifests/split.csv)."""

import csv
from pathlib import Path

MANIFEST_PATH = Path("data/manifests/split.csv")
MANIFEST_FIELDS = ["path", "label", "split", "group", "width", "height", "sha1"]


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest tidak ditemukan: {path}. Jalankan dulu `python -m wastesorter.prepare`."
        )
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
