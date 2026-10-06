"""Metrik klasifikasi yang dipakai di semua tahap."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


@dataclass(frozen=True)
class ClassificationSummary:
    labels: tuple[str, ...]
    accuracy: float
    macro_f1: float
    f1_per_class: dict[str, float]
    confusion: np.ndarray  # baris = label benar, kolom = prediksi


def summarize(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> ClassificationSummary:
    labels = tuple(labels)
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return ClassificationSummary(
        labels=labels,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(per_class.mean()),
        f1_per_class=dict(zip(labels, per_class.tolist())),
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
    )
