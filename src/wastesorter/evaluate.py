"""Tahap 2: pilih ambang "tidak yakin" di validasi, lalu nilai model di test set.

Jalankan dari akar repo:  python -m wastesorter.evaluate
Test set hanya untuk angka akhir: jangan dipakai untuk memilih model atau ambang.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wastesorter.features import eval_transform
from wastesorter.manifest import MANIFEST_PATH, read_manifest
from wastesorter.metrics import ClassificationSummary, summarize
from wastesorter.model import CHECKPOINT_PATH, load_checkpoint
from wastesorter.report import markdown_table
from wastesorter.train import LabeledImages, predict_probabilities

IMAGE_ROOT = Path("data/raw/original")
REPORT_PATH = Path("reports/model.md")
METRICS_PATH = Path("reports/metrics/model.json")
BASELINE_METRICS_PATH = Path("reports/metrics/baseline.json")

# Ambang dipilih sebagai nilai terendah yang membuat akurasi jawaban >= target ini
TARGET_ANSWERED_ACCURACY = 0.98
THRESHOLD_GRID = np.round(np.arange(0.0, 1.0, 0.01), 2)
TABLE_THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)
# Kriteria sukses dari docs/PLAN.md
TARGET_MACRO_F1 = 0.90
TARGET_WEAKEST_F1 = 0.80
BATCH_SIZE = 64


@dataclass(frozen=True)
class Selective:
    threshold: float
    coverage: float  # bagian gambar yang dijawab (yakin >= ambang)
    accuracy: float  # akurasi di antara yang dijawab


def selective(confidences: np.ndarray, correct: np.ndarray, threshold: float) -> Selective:
    answered = confidences >= threshold
    accuracy = float(correct[answered].mean()) if answered.any() else 0.0
    return Selective(float(threshold), float(answered.mean()), accuracy)


def choose_threshold(
    confidences: np.ndarray, correct: np.ndarray, target_accuracy: float
) -> Selective:
    """Ambang terendah yang mencapai target; bila tak ada, yang akurasinya tertinggi."""
    candidates = [selective(confidences, correct, value) for value in THRESHOLD_GRID]
    candidates = [candidate for candidate in candidates if candidate.coverage > 0]
    reaching = [candidate for candidate in candidates if candidate.accuracy >= target_accuracy]
    if reaching:
        return reaching[0]
    return max(candidates, key=lambda candidate: candidate.accuracy)


@dataclass(frozen=True)
class SplitResult:
    summary: ClassificationSummary
    confidences: np.ndarray
    correct: np.ndarray


def evaluate_split(
    model: torch.nn.Module, labels: list[str], rows: list[dict], image_root: Path, workers: int
) -> SplitResult:
    index_of = {label: index for index, label in enumerate(labels)}
    targets = [index_of[row["label"]] for row in rows]
    dataset = LabeledImages([image_root / row["path"] for row in rows], targets, eval_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=workers)
    probabilities = predict_probabilities(model, loader, torch.device("cpu")).numpy()
    predicted = probabilities.argmax(axis=1)
    summary = summarize(
        [row["label"] for row in rows], [labels[index] for index in predicted], labels
    )
    return SplitResult(summary, probabilities.max(axis=1), predicted == np.array(targets))


def _selective_rows(result: SplitResult, chosen: float) -> list[list]:
    thresholds = sorted({*TABLE_THRESHOLDS, chosen})
    rows = []
    for value in thresholds:
        point = selective(result.confidences, result.correct, value)
        name = f"**{value:.2f} (terpilih)**" if value == chosen else f"{value:.2f}"
        rows.append([name, f"{point.coverage:.1%}", f"{point.accuracy:.3f}"])
    return rows


def _check(value: float, target: float) -> str:
    return f"{value:.3f} (target ≥ {target:.2f}: {'tercapai' if value >= target else 'TIDAK tercapai'})"


def write_report(
    path: Path, val: SplitResult, test: SplitResult, threshold: Selective, extra: dict, baseline: dict | None
) -> None:
    labels = test.summary.labels
    weakest = min(labels, key=lambda label: test.summary.f1_per_class[label])
    comparison = [
        ["MobileNetV3 fine-tune", f"{val.summary.accuracy:.3f}", f"{val.summary.macro_f1:.3f}"]
    ]
    if baseline:
        comparison.insert(
            0,
            [
                "Baseline (fitur beku + regresi logistik)",
                f"{baseline['val_accuracy']:.3f}",
                f"{baseline['val_macro_f1']:.3f}",
            ],
        )
    sections = [
        "# Laporan Model (Tahap 2)",
        "Dihasilkan oleh `python -m wastesorter.evaluate`. Jangan diedit manual.",
        f"MobileNetV3-Large fine-tune, checkpoint epoch {extra.get('epoch', '?')} (macro-F1 "
        "validasi terbaik). Model dan ambang dipilih dari validasi saja.",
        "## Hasil di test set",
        markdown_table(
            ["Metrik", "Nilai"],
            [
                ["Jumlah gambar", len(test.correct)],
                ["Akurasi", f"{test.summary.accuracy:.3f}"],
                ["Macro-F1", _check(test.summary.macro_f1, TARGET_MACRO_F1)],
                [f"F1 kelas terlemah ({weakest})", _check(test.summary.f1_per_class[weakest], TARGET_WEAKEST_F1)],
            ],
        ),
        "## Perbandingan dengan baseline (validasi)",
        markdown_table(["Model", "Akurasi", "Macro-F1"], comparison),
        "## F1 per kelas (test)",
        markdown_table(
            ["Kelas", "F1", "Jumlah"],
            [
                [label, f"{test.summary.f1_per_class[label]:.3f}", int(total)]
                for label, total in zip(labels, test.summary.confusion.sum(axis=1))
            ],
        ),
        "## Confusion matrix (test)",
        "Baris = label benar, kolom = prediksi.",
        markdown_table(
            ["", *labels], [[label, *row] for label, row in zip(labels, test.summary.confusion.tolist())]
        ),
        '## Ambang "tidak yakin"',
        f"Ambang = nilai terendah di validasi yang membuat akurasi jawaban ≥ "
        f"{TARGET_ANSWERED_ACCURACY:.0%}. Terpilih: **{threshold.threshold:.2f}**. Di bawah ambang, "
        'app menjawab "tidak yakin".',
        "Validasi:",
        markdown_table(["Ambang", "Dijawab", "Akurasi jawaban"], _selective_rows(val, threshold.threshold)),
        "Test (ambang tidak diubah):",
        markdown_table(["Ambang", "Dijawab", "Akurasi jawaban"], _selective_rows(test, threshold.threshold)),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    model, labels, extra = load_checkpoint(args.checkpoint)
    rows = read_manifest(args.manifest)
    results = {
        split: evaluate_split(
            model, labels, [row for row in rows if row["split"] == split], args.image_root, args.workers
        )
        for split in ("val", "test")
    }
    val, test = results["val"], results["test"]
    threshold = choose_threshold(val.confidences, val.correct, TARGET_ANSWERED_ACCURACY)
    baseline = (
        json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
        if BASELINE_METRICS_PATH.is_file()
        else None
    )

    write_report(REPORT_PATH, val, test, threshold, extra, baseline)
    test_at_threshold = selective(test.confidences, test.correct, threshold.threshold)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "labels": labels,
                "threshold": threshold.threshold,
                "val_accuracy": val.summary.accuracy,
                "val_macro_f1": val.summary.macro_f1,
                "test_accuracy": test.summary.accuracy,
                "test_macro_f1": test.summary.macro_f1,
                "test_f1_per_class": test.summary.f1_per_class,
                "test_coverage_at_threshold": test_at_threshold.coverage,
                "test_accuracy_at_threshold": test_at_threshold.accuracy,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Test: akurasi {test.summary.accuracy:.3f}, macro-F1 {test.summary.macro_f1:.3f}, "
        f"ambang {threshold.threshold:.2f} -> {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
