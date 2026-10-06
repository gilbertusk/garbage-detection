"""Tahap 2: fine-tune MobileNetV3-Large; checkpoint terbaik dipilih dari macro-F1 validasi.

Jalankan dari akar repo:  python -m wastesorter.train
"""

import argparse
import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from wastesorter.features import eval_transform, load_rgb
from wastesorter.manifest import MANIFEST_PATH, read_manifest
from wastesorter.metrics import summarize
from wastesorter.model import (
    CHECKPOINT_PATH,
    Augmentation,
    build_classifier,
    save_checkpoint,
    train_transform,
)

IMAGE_ROOT = Path("data/raw/original")
CONFIG_PATH = Path("configs/train.json")
HISTORY_PATH = Path("reports/metrics/training_history.json")


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    label_smoothing: float
    crop_scale_min: float
    rotation_degrees: float
    color_jitter: float
    workers: int

    @property
    def augmentation(self) -> Augmentation:
        return Augmentation(self.crop_scale_min, self.rotation_degrees, self.color_jitter)


def load_config(path: Path) -> TrainConfig:
    return TrainConfig(**json.loads(path.read_text(encoding="utf-8")))


class LabeledImages(Dataset):
    def __init__(self, paths: list[Path], targets: list[int], transform: Callable) -> None:
        self.paths = paths
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.transform(load_rgb(self.paths[index])), self.targets[index]


def class_weights(targets: list[int], num_classes: int) -> torch.Tensor:
    """Bobot berbanding terbalik dengan frekuensi kelas, dinormalkan ke rata-rata 1."""
    counts = Counter(targets)
    weights = torch.tensor([1.0 / counts[index] for index in range(num_classes)])
    return weights / weights.mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> float:
    model.train()
    total_loss, seen = 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(targets)
        seen += len(targets)
    return total_loss / seen


@torch.no_grad()
def predict_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    outputs = [torch.softmax(model(images.to(device)), dim=1).cpu() for images, _ in loader]
    return torch.cat(outputs)


def _loaders(rows: list[dict], labels: list[str], image_root: Path, config: TrainConfig) -> dict:
    index_of = {label: index for index, label in enumerate(labels)}
    transforms_by_split = {"train": train_transform(config.augmentation), "val": eval_transform()}
    loaders = {}
    for split, transform in transforms_by_split.items():
        split_rows = [row for row in rows if row["split"] == split]
        dataset = LabeledImages(
            [image_root / row["path"] for row in split_rows],
            [index_of[row["label"]] for row in split_rows],
            transform,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.workers,
            persistent_workers=config.workers > 0,
        )
    return loaders


def train(
    rows: list[dict],
    image_root: Path,
    config: TrainConfig,
    checkpoint_path: Path,
    pretrained: bool = True,
) -> list[dict]:
    """Latih model, simpan checkpoint terbaik, kembalikan riwayat per epoch."""
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = sorted({row["label"] for row in rows})
    loaders = _loaders(rows, labels, image_root, config)
    train_targets = loaders["train"].dataset.targets
    val_labels = [labels[target] for target in loaders["val"].dataset.targets]

    model = build_classifier(len(labels), pretrained=pretrained).to(device)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights(train_targets, len(labels)).to(device),
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.learning_rate, total_steps=config.epochs * len(loaders["train"])
    )

    history, best_macro_f1 = [], -1.0
    for epoch in range(1, config.epochs + 1):
        started = time.time()
        loss = train_one_epoch(model, loaders["train"], loss_fn, optimizer, scheduler, device)
        predicted = predict_probabilities(model, loaders["val"], device).argmax(dim=1).tolist()
        summary = summarize(val_labels, [labels[index] for index in predicted], labels)
        entry = {
            "epoch": epoch,
            "train_loss": round(loss, 4),
            "val_accuracy": round(summary.accuracy, 4),
            "val_macro_f1": round(summary.macro_f1, 4),
            "minutes": round((time.time() - started) / 60, 1),
        }
        history.append(entry)
        print(entry, flush=True)
        if summary.macro_f1 > best_macro_f1:
            best_macro_f1 = summary.macro_f1
            save_checkpoint(
                checkpoint_path, model, labels, {"epoch": epoch, "config": asdict(config)}
            )
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    config = load_config(args.config)
    history = train(read_manifest(args.manifest), args.image_root, config, args.checkpoint)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    best = max(history, key=lambda entry: entry["val_macro_f1"])
    print(f"Terbaik: epoch {best['epoch']}, macro-F1 validasi {best['val_macro_f1']}")


if __name__ == "__main__":
    main()
