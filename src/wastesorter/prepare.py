"""Tahap 0: validasi, dedup, split, lalu tulis manifest dan laporan data.

Jalankan dari akar repo:  python -m wastesorter.prepare
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from wastesorter.data import ImageRecord, scan_images
from wastesorter.dedup import group_near_duplicates
from wastesorter.manifest import MANIFEST_PATH, write_manifest
from wastesorter.report import write_report
from wastesorter.split import assign_splits

IMAGE_ROOT = Path("data/raw/original")
REPORT_DIR = Path("reports")

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42
# Korelasi thumbnail minimum agar dua gambar dianggap "hampir sama".
# Dikalibrasi dengan melihat pasangan di sekitar ambang; lihat reports/data.md.
NEAR_DUPLICATE_MIN_CORRELATION = 0.97


@dataclass(frozen=True)
class ExactDuplicates:
    kept: list[ImageRecord]
    dropped_copies: list[str]  # salinan byte-identik dengan label sama
    label_conflicts: list[str]  # file byte-identik yang berada di kelas berbeda


def drop_exact_duplicates(records: list[ImageRecord]) -> ExactDuplicates:
    """Sisakan satu file per sha1; buang semua salinan bila labelnya bertentangan."""
    by_sha1 = defaultdict(list)
    for record in records:
        by_sha1[record.sha1].append(record)

    kept, dropped, conflicts = [], [], []
    for copies in by_sha1.values():
        if len({record.label for record in copies}) > 1:
            conflicts.extend(record.path for record in copies)
            continue
        kept.append(copies[0])
        dropped.extend(record.path for record in copies[1:])
    return ExactDuplicates(sorted(kept, key=lambda r: r.path), sorted(dropped), sorted(conflicts))


def build_manifest(records: list[ImageRecord]) -> list[dict]:
    signatures = [record.signature for record in records]
    group_ids = group_near_duplicates(signatures, NEAR_DUPLICATE_MIN_CORRELATION).tolist()
    labels = [record.label for record in records]
    splits = assign_splits(labels, group_ids, SPLIT_RATIOS, SPLIT_SEED)
    return [
        {
            "path": record.path,
            "label": record.label,
            "split": split,
            "group": f"g{group_id:05d}",
            "width": record.width,
            "height": record.height,
            "sha1": record.sha1,
        }
        for record, group_id, split in zip(records, group_ids, splits)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    if not args.image_root.is_dir():
        raise SystemExit(f"Folder gambar tidak ditemukan: {args.image_root}")

    records, invalid = scan_images(args.image_root)
    if not records:
        raise SystemExit(f"Tidak ada gambar yang terbaca di {args.image_root}")

    exact = drop_exact_duplicates(records)
    rows = build_manifest(exact.kept)
    write_manifest(rows, args.manifest)
    write_report(
        rows=rows,
        scanned=len(records) + len(invalid),
        invalid=invalid,
        dropped_copies=exact.dropped_copies,
        label_conflicts=exact.label_conflicts,
        image_root=args.image_root,
        report_dir=args.report_dir,
        ratios=SPLIT_RATIOS,
        min_correlation=NEAR_DUPLICATE_MIN_CORRELATION,
        seed=SPLIT_SEED,
    )
    print(f"{len(rows)} gambar -> {args.manifest}; laporan di {args.report_dir / 'data.md'}")


if __name__ == "__main__":
    main()
