import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from wastesorter.export import (
    _quantize_per_channel,
    export_onnx,
    onnx_logits,
    onnx_session,
    quantize_weights,
    write_model_config,
)
from wastesorter.features import CROP_SIZE, IMAGENET_MEAN, IMAGENET_STD
from wastesorter.model import build_classifier

PREPROCESS_FIXTURE = Path("web/tests/fixtures/preprocess.json")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    model = build_classifier(3, pretrained=False).eval()
    path = tmp_path_factory.mktemp("onnx") / "float32.onnx"
    export_onnx(model, path)
    return model, path


def test_onnx_model_matches_pytorch_outputs(exported):
    model, path = exported
    batch = torch.randn(2, 3, CROP_SIZE, CROP_SIZE)

    with torch.no_grad():
        expected = model(batch).numpy()
    actual = onnx_logits(onnx_session(path), batch.numpy())

    assert actual.shape == (2, 3)
    assert np.abs(actual - expected).max() < 1e-3


def test_weight_quantized_model_is_much_smaller_and_stays_close(exported):
    _, float_path = exported
    quantized_path = float_path.with_name("int8.onnx")
    batch = np.random.default_rng(0).normal(size=(4, 3, CROP_SIZE, CROP_SIZE)).astype(np.float32)

    quantize_weights(float_path, quantized_path)

    expected = onnx_logits(onnx_session(float_path), batch)
    actual = onnx_logits(onnx_session(quantized_path), batch)
    assert quantized_path.stat().st_size < float_path.stat().st_size / 3
    assert np.abs(actual - expected).max() < 0.1 * np.abs(expected).max()


def test_per_channel_quantization_reconstructs_weights_within_half_a_step():
    weights = np.random.default_rng(0).normal(size=(4, 3, 3, 3)).astype(np.float32)
    weights[1] = 0.0  # kanal kosong tidak boleh menghasilkan pembagian nol

    quantized, scale = _quantize_per_channel(weights)

    restored = quantized.astype(np.float32) * scale.reshape(-1, 1, 1, 1)
    assert quantized.dtype == np.int8
    assert np.abs(restored - weights).max() <= scale.max() / 2 + 1e-6


def test_model_config_carries_what_the_app_needs(tmp_path):
    write_model_config(tmp_path / "model.json", ["glass", "metal"], threshold=0.6)

    config = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))

    assert config["labels"] == ["glass", "metal"]
    assert config["threshold"] == 0.6
    assert config["inputSize"] == 224
    assert config["cropFraction"] == pytest.approx(0.875)
    assert config["mean"] == pytest.approx(list(IMAGENET_MEAN))


def test_javascript_preprocess_fixture_matches_torchvision():
    """Fixture yang sama diuji oleh web/tests/preprocess.test.js terhadap kode JavaScript."""
    fixture = json.loads(PREPROCESS_FIXTURE.read_text(encoding="utf-8"))
    side = fixture["side"]
    rgba = np.array(fixture["rgba"], dtype=np.uint8).reshape(side, side, 4)
    to_tensor = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )

    expected = to_tensor(Image.fromarray(rgba[:, :, :3])).flatten().tolist()

    assert fixture["expected"] == pytest.approx(expected, abs=1e-5)
