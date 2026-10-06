import numpy as np
import pytest
import torch
from PIL import Image

from wastesorter.export import export_onnx, write_model_config
from wastesorter.features import eval_transform
from wastesorter.model import build_classifier
from wastesorter.predict import WasteClassifier, preprocess

CONFIG = {
    "inputSize": 224,
    "cropFraction": 0.875,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


@pytest.mark.parametrize("size", [(640, 480), (480, 640), (300, 300)])
def test_preprocess_matches_the_training_transform(tmp_path, make_image, size):
    image = Image.open(make_image(tmp_path / "a.jpg", seed=1, size=size)).convert("RGB")

    actual = preprocess(image, CONFIG)

    expected = eval_transform()(image).numpy()[None]
    assert actual.shape == expected.shape
    assert np.abs(actual - expected).max() < 1e-4


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    torch.manual_seed(0)
    directory = tmp_path_factory.mktemp("model")
    export_onnx(build_classifier(3, pretrained=False), directory / "waste.onnx")
    write_model_config(directory / "model.json", ["glass", "metal", "plastic"], threshold=0.99)
    return directory


def test_prediction_lists_guesses_from_most_to_least_likely(model_dir, tmp_path, make_image):
    photo = make_image(tmp_path / "a.jpg", seed=1, size=(320, 240))

    prediction = WasteClassifier(model_dir).predict(photo)

    confidences = [confidence for _, confidence in prediction.top]
    assert {label for label, _ in prediction.top} == {"glass", "metal", "plastic"}
    assert confidences == sorted(confidences, reverse=True)
    assert sum(confidences) == pytest.approx(1.0, abs=1e-5)


def test_prediction_below_the_threshold_is_not_confident(model_dir, tmp_path, make_image):
    photo = make_image(tmp_path / "a.jpg", seed=1)

    assert WasteClassifier(model_dir).predict(photo).is_confident is False


def test_missing_model_explains_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="wastesorter.export"):
        WasteClassifier(tmp_path)
