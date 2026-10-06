import pytest
import torch

from wastesorter.model import load_checkpoint
from wastesorter.train import CONFIG_PATH, TrainConfig, class_weights, load_config, train

TINY_CONFIG = TrainConfig(
    seed=0,
    epochs=1,
    batch_size=4,
    learning_rate=1e-3,
    weight_decay=0.0,
    label_smoothing=0.0,
    crop_scale_min=0.8,
    rotation_degrees=5,
    color_jitter=0.1,
    workers=0,
)


def test_class_weights_favor_the_rare_class_and_average_one():
    weights = class_weights([0, 0, 0, 1], num_classes=2)

    assert weights[1] == pytest.approx(3 * weights[0])
    assert weights.mean() == pytest.approx(1.0)


def test_repo_config_loads():
    config = load_config(CONFIG_PATH)

    assert config.epochs > 0
    assert 0 < config.crop_scale_min <= 1


def test_training_writes_a_checkpoint_that_can_be_reloaded(tmp_path, make_image):
    rows = []
    for label_index, label in enumerate(("glass", "metal")):
        for index in range(6):
            name = f"{label}/{index}.jpg"
            make_image(tmp_path / "images" / name, seed=label_index * 100 + index)
            rows.append({"path": name, "label": label, "split": "train" if index < 4 else "val"})
    checkpoint = tmp_path / "model" / "best.pt"

    history = train(rows, tmp_path / "images", TINY_CONFIG, checkpoint, pretrained=False)

    model, labels, extra = load_checkpoint(checkpoint)
    assert [entry["epoch"] for entry in history] == [1]
    assert labels == ["glass", "metal"]
    assert extra["epoch"] == 1
    assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 2)


def test_loading_a_missing_checkpoint_explains_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="wastesorter.train"):
        load_checkpoint(tmp_path / "best.pt")
