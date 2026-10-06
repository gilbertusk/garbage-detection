"""Tahap 1: baseline fitur MobileNetV3 beku + regresi logistik, dinilai di validasi.

Jalankan dari akar repo:  python -m wastesorter.baseline
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from wastesorter.features import build_backbone, extract_features
from wastesorter.manifest import MANIFEST_PATH, read_manifest
from wastesorter.metrics import ClassificationSummary, summarize
from wastesorter.report import markdown_table

IMAGE_ROOT = Path("data/raw/original")
FEATURE_CACHE = Path("runs/features")
REPORT_PATH = Path("reports/baseline.md")
METRICS_PATH = Path("reports/metrics/baseline.json")

REGULARIZATION_C = 1.0
MAX_ITERATIONS = 2000


def fit_logistic(features: np.ndarray, labels: list[str]) -> Pipeline:
    """Standardisasi + regresi logistik dengan bobot kelas seimbang."""
    classifier = LogisticRegression(
        C=REGULARIZATION_C, max_iter=MAX_ITERATIONS, class_weight="balanced"
    )
    return make_pipeline(StandardScaler(), classifier).fit(features, labels)


def _cached_features(rows: list[dict], split: str, image_root: Path, cache_dir: Path) -> np.ndarray:
    paths = [row["path"] for row in rows]
    cache_file = cache_dir / f"{split}.npz"
    if cache_file.is_file():
        cached = np.load(cache_file)
        if cached["paths"].tolist() == paths:
            return cached["features"]
    features = extract_features([image_root / path for path in paths], build_backbone())
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, paths=np.array(paths), features=features)
    return features


def _write_report(
    path: Path, model: ClassificationSummary, majority: ClassificationSummary, counts: dict
) -> None:
    labels = model.labels
    confusion_rows = [[label, *row] for label, row in zip(labels, model.confusion.tolist())]
    sections = [
        "# Laporan Baseline (Tahap 1)",
        "Dihasilkan oleh `python -m wastesorter.baseline`. Jangan diedit manual.",
        "Fitur 960-dimensi dari MobileNetV3-Large beku (ImageNet), lalu standardisasi dan "
        f"regresi logistik (C={REGULARIZATION_C}, bobot kelas seimbang). Dilatih pada "
        f"{counts['train']} gambar train, dinilai pada {counts['val']} gambar validasi. "
        "Test set tidak disentuh.",
        "## Hasil di validasi",
        markdown_table(
            ["Model", "Akurasi", "Macro-F1"],
            [
                ["Selalu tebak kelas terbanyak", f"{majority.accuracy:.3f}", f"{majority.macro_f1:.3f}"],
                ["**MobileNetV3 beku + regresi logistik**", f"**{model.accuracy:.3f}**", f"**{model.macro_f1:.3f}**"],
            ],
        ),
        "## F1 per kelas",
        markdown_table(
            ["Kelas", "F1", "Jumlah di validasi"],
            [
                [label, f"{model.f1_per_class[label]:.3f}", int(row_total)]
                for label, row_total in zip(labels, model.confusion.sum(axis=1))
            ],
        ),
        "## Confusion matrix",
        "Baris = label benar, kolom = prediksi.",
        markdown_table(["", *labels], confusion_rows),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    by_split = {
        split: [row for row in rows if row["split"] == split] for split in ("train", "val")
    }
    features = {
        split: _cached_features(split_rows, split, args.image_root, FEATURE_CACHE)
        for split, split_rows in by_split.items()
    }
    train_labels = [row["label"] for row in by_split["train"]]
    val_labels = [row["label"] for row in by_split["val"]]
    labels = sorted(set(train_labels))

    predictions = fit_logistic(features["train"], train_labels).predict(features["val"])
    model = summarize(val_labels, predictions.tolist(), labels)
    most_common = Counter(train_labels).most_common(1)[0][0]
    majority = summarize(val_labels, [most_common] * len(val_labels), labels)

    counts = {split: len(split_rows) for split, split_rows in by_split.items()}
    _write_report(args.report, model, majority, counts)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps({"val_accuracy": model.accuracy, "val_macro_f1": model.macro_f1}, indent=2),
        encoding="utf-8",
    )
    print(f"Validasi: akurasi {model.accuracy:.3f}, macro-F1 {model.macro_f1:.3f} -> {args.report}")


if __name__ == "__main__":
    main()
