import numpy as np
import pytest

from wastesorter.baseline import fit_logistic
from wastesorter.features import FEATURE_DIM, build_backbone, extract_features
from wastesorter.manifest import read_manifest
from wastesorter.metrics import summarize


def test_summary_reports_accuracy_and_macro_f1():
    y_true = ["glass", "glass", "metal", "metal"]
    y_pred = ["glass", "metal", "metal", "metal"]

    summary = summarize(y_true, y_pred, labels=["glass", "metal"])

    assert summary.accuracy == pytest.approx(0.75)
    assert summary.f1_per_class["glass"] == pytest.approx(2 / 3)
    assert summary.f1_per_class["metal"] == pytest.approx(0.8)
    assert summary.macro_f1 == pytest.approx((2 / 3 + 0.8) / 2)
    assert summary.confusion.tolist() == [[1, 1], [0, 2]]


def test_summary_counts_a_never_predicted_class_as_zero_f1():
    summary = summarize(["glass", "trash"], ["glass", "glass"], labels=["glass", "trash"])

    assert summary.f1_per_class["trash"] == 0.0


def test_logistic_baseline_separates_well_separated_features():
    rng = np.random.default_rng(0)
    glass = rng.normal(loc=-3.0, size=(40, 5))
    metal = rng.normal(loc=3.0, size=(40, 5))
    features = np.vstack([glass, metal])
    labels = ["glass"] * 40 + ["metal"] * 40

    model = fit_logistic(features, labels)

    assert model.predict(np.full((1, 5), -3.0)).tolist() == ["glass"]
    assert model.predict(np.full((1, 5), 3.0)).tolist() == ["metal"]


def test_backbone_gives_one_feature_vector_per_image(tmp_path, make_image):
    paths = [make_image(tmp_path / f"{seed}.jpg", seed=seed, size=(300, 200)) for seed in range(3)]

    features = extract_features(paths, build_backbone(pretrained=False), workers=0)

    assert features.shape == (3, FEATURE_DIM)
    assert np.isfinite(features).all()


def test_reading_a_missing_manifest_explains_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="wastesorter.prepare"):
        read_manifest(tmp_path / "split.csv")
