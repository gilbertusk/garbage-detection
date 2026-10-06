"""Pakai model hasil ekspor di luar browser: hanya butuh onnxruntime, numpy, dan Pillow.

    python -m wastesorter.predict foto.jpg

Atau dari kode lain:

    from wastesorter.predict import WasteClassifier
    classifier = WasteClassifier("web/model")
    prediction = classifier.predict("foto.jpg")
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

MODEL_DIR = Path("web/model")
CONFIG_FILE = "model.json"
TOP_GUESSES = 3
MAX_PIXEL = 255.0


@dataclass(frozen=True)
class Prediction:
    is_confident: bool  # False berarti "tidak yakin": jangan dipakai sebagai jawaban
    top: tuple[tuple[str, float], ...]  # (label, kemungkinan), dari yang paling yakin


def preprocess(image: Image.Image, config: dict) -> np.ndarray:
    """Sama dengan eval_transform() saat training: sisi pendek -> crop tengah -> normalisasi."""
    size = config["inputSize"]
    resize_to = round(size / config["cropFraction"])
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = resize_to / min(image.size)
    resized = image.resize(
        (max(resize_to, int(image.width * scale)), max(resize_to, int(image.height * scale))),
        Image.Resampling.BILINEAR,
    )
    left = round((resized.width - size) / 2)
    top = round((resized.height - size) / 2)
    pixels = np.asarray(resized.crop((left, top, left + size, top + size)), dtype=np.float32)
    normalized = (pixels / MAX_PIXEL - np.array(config["mean"])) / np.array(config["std"])
    return normalized.transpose(2, 0, 1)[None].astype(np.float32)


class WasteClassifier:
    def __init__(self, model_dir: Path | str = MODEL_DIR) -> None:
        model_dir = Path(model_dir)
        config_path = model_dir / CONFIG_FILE
        if not config_path.is_file():
            raise FileNotFoundError(
                f"{config_path} tidak ditemukan. Jalankan dulu `python -m wastesorter.export`."
            )
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.session = ort.InferenceSession(
            str(model_dir / self.config["modelFile"]), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: Image.Image | Path | str) -> Prediction:
        if not isinstance(image, Image.Image):
            with Image.open(image) as opened:
                return self.predict(opened.copy())
        logits = self.session.run(None, {self.input_name: preprocess(image, self.config)})[0][0]
        exponents = np.exp(logits - logits.max())
        probabilities = exponents / exponents.sum()
        order = np.argsort(-probabilities)[:TOP_GUESSES]
        top = tuple((self.config["labels"][index], float(probabilities[index])) for index in order)
        return Prediction(is_confident=top[0][1] >= self.config["threshold"], top=top)


def main() -> None:
    parser = argparse.ArgumentParser(description="Klasifikasikan foto sampah dengan model ONNX.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    classifier = WasteClassifier(args.model_dir)
    for path in args.images:
        prediction = classifier.predict(path)
        guesses = ", ".join(f"{label} {confidence:.0%}" for label, confidence in prediction.top)
        verdict = prediction.top[0][0] if prediction.is_confident else "tidak yakin"
        print(f"{path}: {verdict} ({guesses})")


if __name__ == "__main__":
    main()
