import numpy as np
import pytest
from PIL import Image

from wastesorter.features import load_rgb
from wastesorter.real_world import NON_WASTE_TARGET, analyze, collect_photos

LABELS = ["glass", "metal", "plastic"]


def test_photos_get_their_class_from_the_folder_name(tmp_path, make_image):
    make_image(tmp_path / "metal" / "a.jpg", seed=1)
    make_image(tmp_path / "bukan_sampah" / "hand.jpg", seed=2)
    make_image(tmp_path / "kaleng" / "typo.jpg", seed=3)

    photos, unknown = collect_photos(tmp_path, LABELS)

    assert [(photo.path.name, photo.target) for photo in photos] == [
        ("hand.jpg", NON_WASTE_TARGET),
        ("a.jpg", 1),
    ]
    assert unknown == ["kaleng"]


def test_missing_photo_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_photos(tmp_path / "real", LABELS)


def test_analysis_separates_waste_accuracy_from_non_waste_rejection():
    probabilities = np.array(
        [
            [0.90, 0.05, 0.05],  # glass, benar, yakin
            [0.20, 0.30, 0.50],  # glass ditebak plastic, ragu
            [0.10, 0.80, 0.10],  # metal, benar, yakin
            [0.40, 0.35, 0.25],  # bukan sampah, ragu -> ditolak
            [0.95, 0.03, 0.02],  # bukan sampah, tetapi model yakin
        ]
    )
    targets = np.array([0, 0, 1, NON_WASTE_TARGET, NON_WASTE_TARGET])

    result = analyze(probabilities, targets, LABELS, threshold=0.7)

    assert result.waste_count == 3
    assert result.accuracy == pytest.approx(2 / 3)
    assert result.top_k_accuracy == pytest.approx(1.0)
    assert result.coverage == pytest.approx(2 / 3)
    assert result.answered_accuracy == pytest.approx(1.0)
    assert result.per_class == {"glass": (1, 2), "metal": (1, 1)}
    assert result.confusions == [("glass", "plastic", 1)]
    assert result.non_waste_count == 2
    assert result.non_waste_rejected == pytest.approx(0.5)


def test_phone_photo_is_rotated_according_to_exif(tmp_path):
    # HP menyimpan foto potret sebagai piksel lanskap + tag orientasi EXIF
    image = Image.new("RGB", (40, 20), "red")
    exif = Image.Exif()
    exif[0x0112] = 6  # putar 90 derajat
    image.save(tmp_path / "portrait.jpg", exif=exif)

    assert load_rgb(tmp_path / "portrait.jpg").size == (20, 40)
