import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def make_image():
    """Buat gambar acak yang halus (bukan noise murni) agar tahan resize."""

    def _make(path, seed, size=(96, 96)):
        rng = np.random.default_rng(seed)
        small = rng.integers(0, 256, size=(6, 6, 3), dtype=np.uint8)
        image = Image.fromarray(small).resize(size, Image.Resampling.BICUBIC)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, quality=95)
        return path

    return _make
