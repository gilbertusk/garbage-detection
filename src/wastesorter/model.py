"""Model klasifikasi, augmentasi training, dan baca/tulis checkpoint."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from wastesorter.features import CROP_SIZE, IMAGENET_MEAN, IMAGENET_STD

CHECKPOINT_PATH = Path("runs/model/best.pt")


@dataclass(frozen=True)
class Augmentation:
    crop_scale_min: float
    rotation_degrees: float
    color_jitter: float


def build_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    """MobileNetV3-Large dengan lapisan keluaran diganti menjadi num_classes."""
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    last = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return model


def train_transform(augmentation: Augmentation) -> transforms.Compose:
    """Augmentasi: crop acak, balik horizontal, putar, dan variasi cahaya/warna."""
    jitter = augmentation.color_jitter
    return transforms.Compose(
        [
            transforms.RandomRotation(augmentation.rotation_degrees),
            transforms.RandomResizedCrop(CROP_SIZE, scale=(augmentation.crop_scale_min, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def save_checkpoint(path: Path, model: nn.Module, labels: list[str], extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "labels": labels, **extra}, path)


def load_checkpoint(path: Path) -> tuple[nn.Module, list[str], dict]:
    """Mengembalikan (model dalam mode eval, daftar label, isi checkpoint lainnya)."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {path}. Jalankan dulu `python -m wastesorter.train`."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    labels = checkpoint["labels"]
    model = build_classifier(len(labels), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    extra = {key: value for key, value in checkpoint.items() if key not in ("state_dict", "labels")}
    return model.eval(), labels, extra
