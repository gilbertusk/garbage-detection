"""Laporan data tahap 0: tabel Markdown dan gambar contoh."""

import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from PIL import Image

THUMBNAIL = 128
SAMPLES_PER_CLASS = 8
DUPLICATE_GROUPS_SHOWN = 10
DUPLICATE_GROUP_WIDTH = 4
FIGURE_SEED = 0


def _thumbnail(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((THUMBNAIL, THUMBNAIL))
        canvas = Image.new("RGB", (THUMBNAIL, THUMBNAIL), "white")
        canvas.paste(image, ((THUMBNAIL - image.width) // 2, (THUMBNAIL - image.height) // 2))
        return canvas


def _grid(image_rows: list[list[Path]], columns: int, output: Path) -> None:
    sheet = Image.new("RGB", (columns * THUMBNAIL, len(image_rows) * THUMBNAIL), "white")
    for row_index, paths in enumerate(image_rows):
        for column_index, path in enumerate(paths[:columns]):
            sheet.paste(_thumbnail(path), (column_index * THUMBNAIL, row_index * THUMBNAIL))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=80)


def _sample_figure(rows: list[dict], image_root: Path, output: Path) -> list[str]:
    rng = random.Random(FIGURE_SEED)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(image_root / row["path"])
    labels = sorted(by_label)
    picks = [rng.sample(by_label[label], min(SAMPLES_PER_CLASS, len(by_label[label]))) for label in labels]
    _grid(picks, SAMPLES_PER_CLASS, output)
    return labels


def _duplicate_figure(groups: dict[str, list[dict]], image_root: Path, output: Path) -> int:
    multi = [members for members in groups.values() if len(members) > 1]
    if not multi:
        return 0
    rng = random.Random(FIGURE_SEED)
    shown = rng.sample(multi, min(DUPLICATE_GROUPS_SHOWN, len(multi)))
    _grid(
        [[image_root / row["path"] for row in members] for members in shown],
        DUPLICATE_GROUP_WIDTH,
        output,
    )
    return len(shown)


def markdown_table(header: list[str], body: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in line) + " |" for line in body]
    return "\n".join(lines)


def _split_table(rows: list[dict], ratios: dict[str, float]) -> str:
    counts = Counter((row["label"], row["split"]) for row in rows)
    labels = sorted({row["label"] for row in rows})
    body = []
    for label in labels:
        per_split = [counts[(label, split)] for split in ratios]
        body.append([label, *per_split, sum(per_split)])
    totals = [sum(counts[(label, split)] for label in labels) for split in ratios]
    body.append(["**total**", *totals, sum(totals)])
    return markdown_table(["Kelas", *ratios, "Total"], body)


def write_report(
    *,
    rows: list[dict],
    scanned: int,
    invalid: list[str],
    dropped_copies: list[str],
    label_conflicts: list[str],
    image_root: Path,
    report_dir: Path,
    ratios: dict[str, float],
    min_correlation: float,
    seed: int,
) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    multi = [members for members in groups.values() if len(members) > 1]
    cross_class = [m for m in multi if len({row["label"] for row in m}) > 1]
    leaking = [m for m in multi if len({row["split"] for row in m}) > 1]

    labels = _sample_figure(rows, image_root, report_dir / "figures" / "samples.jpg")
    shown = _duplicate_figure(groups, image_root, report_dir / "figures" / "near_duplicates.jpg")

    widths = [row["width"] for row in rows]
    heights = [row["height"] for row in rows]
    summary = [
        ["File gambar ditemukan", scanned],
        ["Rusak / tidak terbaca", len(invalid)],
        ["Salinan byte-identik dibuang", len(dropped_copies)],
        ["Dibuang karena byte-identik di kelas berbeda", len(label_conflicts)],
        ["**Gambar di manifest**", f"**{len(rows)}**"],
        ["Grup hampir-sama (≥ 2 gambar)", len(multi)],
        ["Gambar di dalam grup tersebut", sum(len(m) for m in multi)],
        ["Grup terbesar", max((len(m) for m in multi), default=1)],
        ["Grup yang melintasi kelas", len(cross_class)],
        ["Grup yang melintasi split", len(leaking)],
    ]
    sections = [
        "# Laporan Data (Tahap 0)",
        f"Dihasilkan oleh `python -m wastesorter.prepare` dari `{image_root.as_posix()}`. "
        "Jangan diedit manual.",
        "## Ringkasan",
        markdown_table(["Hal", "Jumlah"], summary),
        f"Hampir-sama = korelasi thumbnail grayscale 32×32 ≥ {min_correlation}, dikelompokkan "
        "secara transitif. Satu grup selalu masuk utuh ke satu split.",
        "Keterbatasan: kriteria ini menangkap salinan, resize, dan kompresi ulang, tetapi "
        "tidak menangkap benda sama yang difoto ulang dari sudut lain, gambar yang dibalik "
        "atau di-crop, maupun foto berbeda dari sesi pemotretan yang sama (latar identik). "
        "Yang terakhir banyak di dataset ini dan tetap tersebar di ketiga split.",
        "## Split per kelas",
        f"Target {' / '.join(f'{ratio:.0%}' for ratio in ratios.values())}, seed {seed}.",
        _split_table(rows, ratios),
        "## Ukuran gambar",
        markdown_table(
            ["", "min", "median", "maks"],
            [
                ["lebar", min(widths), int(median(widths)), max(widths)],
                ["tinggi", min(heights), int(median(heights)), max(heights)],
            ],
        ),
        "## Contoh gambar",
        f"Satu baris per kelas, urut: {', '.join(labels)}.",
        "![contoh per kelas](figures/samples.jpg)",
    ]
    if shown:
        sections += [
            "## Contoh grup hampir-sama",
            f"{shown} grup acak, satu baris per grup (maks. {DUPLICATE_GROUP_WIDTH} gambar).",
            "![grup hampir-sama](figures/near_duplicates.jpg)",
        ]
    if invalid:
        sections += ["## File rusak", "\n".join(f"- `{path}`" for path in invalid)]

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "data.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
