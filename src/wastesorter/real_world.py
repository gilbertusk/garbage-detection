"""Tahap 3: uji model pada foto HP sendiri di data/real/<kelas>/.

Jalankan dari akar repo:  python -m wastesorter.real_world
Foto benda yang bukan sampah ditaruh di data/real/bukan_sampah/.
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wastesorter.data import IMAGE_SUFFIXES
from wastesorter.features import eval_transform
from wastesorter.metrics import summarize
from wastesorter.model import CHECKPOINT_PATH, load_checkpoint
from wastesorter.report import markdown_table
from wastesorter.train import LabeledImages, predict_probabilities

REAL_ROOT = Path("data/real")
REPORT_PATH = Path("reports/real_world.md")
MODEL_METRICS_PATH = Path("reports/metrics/model.json")
NON_WASTE_FOLDER = "bukan_sampah"
NON_WASTE_TARGET = -1
TOP_K = 3
CONFUSIONS_SHOWN = 10
BATCH_SIZE = 32


@dataclass(frozen=True)
class Photo:
    path: Path
    target: int  # indeks kelas, atau NON_WASTE_TARGET


def collect_photos(root: Path, labels: list[str]) -> tuple[list[Photo], list[str]]:
    """Kumpulkan foto per folder kelas. Mengembalikan (foto, nama folder tak dikenal)."""
    if not root.is_dir():
        raise FileNotFoundError(f"Folder foto tidak ditemukan: {root}")
    targets = {label: index for index, label in enumerate(labels)}
    targets[NON_WASTE_FOLDER] = NON_WASTE_TARGET
    photos, unknown = [], []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        if folder.name not in targets:
            unknown.append(folder.name)
            continue
        photos += [
            Photo(file, targets[folder.name])
            for file in sorted(folder.iterdir())
            if file.suffix.lower() in IMAGE_SUFFIXES
        ]
    return photos, unknown


@dataclass(frozen=True)
class RealWorldResult:
    waste_count: int
    accuracy: float
    top_k_accuracy: float
    macro_f1: float
    coverage: float  # bagian foto sampah yang dijawab (yakin >= ambang)
    answered_accuracy: float
    per_class: dict[str, tuple[int, int]]  # kelas -> (benar, jumlah)
    confusions: list[tuple[str, str, int]]  # (label benar, prediksi, jumlah)
    non_waste_count: int
    non_waste_rejected: float  # bagian foto non-sampah yang dijawab "tidak yakin"


def analyze(
    probabilities: np.ndarray, targets: np.ndarray, labels: list[str], threshold: float
) -> RealWorldResult:
    confidences = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    is_waste = targets != NON_WASTE_TARGET
    waste_targets, waste_predicted = targets[is_waste], predicted[is_waste]
    correct = waste_predicted == waste_targets
    answered = confidences[is_waste] >= threshold
    top_k = np.argsort(-probabilities[is_waste], axis=1)[:, :TOP_K]

    true_names = [labels[index] for index in waste_targets]
    predicted_names = [labels[index] for index in waste_predicted]
    present = sorted(set(true_names))
    wrong_pairs = Counter(
        (true, guess) for true, guess in zip(true_names, predicted_names) if true != guess
    )
    non_waste_confidences = confidences[~is_waste]
    return RealWorldResult(
        waste_count=int(is_waste.sum()),
        accuracy=float(correct.mean()),
        top_k_accuracy=float((top_k == waste_targets[:, None]).any(axis=1).mean()),
        macro_f1=summarize(true_names, predicted_names, present).macro_f1,
        coverage=float(answered.mean()),
        answered_accuracy=float(correct[answered].mean()) if answered.any() else 0.0,
        per_class={
            label: (
                int(correct[waste_targets == labels.index(label)].sum()),
                int((waste_targets == labels.index(label)).sum()),
            )
            for label in present
        },
        confusions=[(true, guess, count) for (true, guess), count in wrong_pairs.most_common(CONFUSIONS_SHOWN)],
        non_waste_count=int((~is_waste).sum()),
        non_waste_rejected=float((non_waste_confidences < threshold).mean())
        if len(non_waste_confidences)
        else 0.0,
    )


def write_report(path: Path, result: RealWorldResult, threshold: float, test: dict | None) -> None:
    summary = [
        ["Foto sampah", result.waste_count],
        ["Akurasi (tebakan teratas)", f"{result.accuracy:.3f}"],
        [f"Akurasi {TOP_K} tebakan teratas", f"{result.top_k_accuracy:.3f}"],
        ["Macro-F1", f"{result.macro_f1:.3f}"],
        [f"Dijawab pada ambang {threshold:.2f}", f"{result.coverage:.1%}"],
        ["Akurasi di antara yang dijawab", f"{result.answered_accuracy:.3f}"],
    ]
    if test:
        gap = test["test_accuracy"] - result.accuracy
        summary += [
            ["Akurasi test set Kaggle (pembanding)", f"{test['test_accuracy']:.3f}"],
            ["**Penurunan dari test ke foto nyata**", f"**{gap * 100:.1f} poin persentase**"],
        ]
    sections = [
        "# Laporan Uji Dunia Nyata (Tahap 3)",
        "Dihasilkan oleh `python -m wastesorter.real_world`. Jangan diedit manual.",
        "Foto diambil dengan HP dan dilabeli sendiri; tidak pernah dipakai untuk training "
        "maupun memilih ambang. Jumlahnya kecil, jadi angka per kelas bersifat kasar.",
        "## Ringkasan",
        markdown_table(["Hal", "Nilai"], summary),
        "## Per kelas",
        markdown_table(
            ["Kelas", "Benar", "Jumlah"],
            [[label, right, total] for label, (right, total) in result.per_class.items()],
        ),
    ]
    if result.confusions:
        sections += [
            "## Yang paling sering tertukar",
            markdown_table(["Label benar", "Ditebak sebagai", "Jumlah"], [list(row) for row in result.confusions]),
        ]
    if result.non_waste_count:
        sections += [
            "## Foto bukan sampah",
            f"{result.non_waste_count} foto; {result.non_waste_rejected:.1%} dijawab "
            f'"tidak yakin" (yakin < {threshold:.2f}). Sisanya ditebak sebagai sampah dengan yakin.',
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real-root", type=Path, default=REAL_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    if not MODEL_METRICS_PATH.is_file():
        raise SystemExit(f"{MODEL_METRICS_PATH} belum ada. Jalankan dulu `python -m wastesorter.evaluate`.")
    test_metrics = json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    model, labels, _ = load_checkpoint(args.checkpoint)

    photos, unknown = collect_photos(args.real_root, labels)
    if unknown:
        print(f"Folder dilewati (bukan nama kelas): {', '.join(unknown)}")
    if not any(photo.target != NON_WASTE_TARGET for photo in photos):
        raise SystemExit(f"Belum ada foto sampah di {args.real_root}/<kelas>/.")

    targets = [photo.target for photo in photos]
    dataset = LabeledImages([photo.path for photo in photos], targets, eval_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    probabilities = predict_probabilities(model, loader, torch.device("cpu")).numpy()

    threshold = test_metrics["threshold"]
    result = analyze(probabilities, np.array(targets), labels, threshold)
    write_report(REPORT_PATH, result, threshold, test_metrics)
    print(f"Foto nyata: akurasi {result.accuracy:.3f} dari {result.waste_count} foto -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
