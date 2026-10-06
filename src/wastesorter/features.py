"""Ekstraksi fitur dari MobileNetV3-Large beku (pretrained ImageNet)."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

RESIZE_SIZE = 256
CROP_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FEATURE_DIM = 960


def eval_transform() -> transforms.Compose:
    """Preprocessing deterministik: sisi pendek 256 -> crop tengah 224 -> normalisasi."""
    return transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_backbone(pretrained: bool = True) -> torch.nn.Module:
    """MobileNetV3-Large tanpa kepala klasifikasi; keluaran FEATURE_DIM per gambar."""
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    model.classifier = torch.nn.Identity()
    return model.eval()


def load_rgb(path: Path) -> Image.Image:
    """Buka gambar sebagai RGB tegak; JPEG besar di-decode di skala kecil (draft)."""
    with Image.open(path) as image:
        image.draft("RGB", (RESIZE_SIZE * 2, RESIZE_SIZE * 2))
        # Foto HP menyimpan orientasi di EXIF; tanpa ini foto potret terbaca miring
        return ImageOps.exif_transpose(image).convert("RGB")


class _ImageDataset(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.transform = eval_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(load_rgb(self.paths[index]))


@torch.no_grad()
def extract_features(
    paths: list[Path], model: torch.nn.Module, batch_size: int = 64, workers: int = 4
) -> np.ndarray:
    loader = DataLoader(_ImageDataset(paths), batch_size=batch_size, num_workers=workers)
    batches = [model(batch).numpy() for batch in loader]
    return np.concatenate(batches).astype(np.float32)
