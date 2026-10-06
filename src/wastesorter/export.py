"""Tahap 4: ekspor PyTorch -> ONNX -> bobot int8, dengan uji paritas dan akurasi.

Jalankan dari akar repo:  python -m wastesorter.export
Menulis web/model/waste.onnx, web/model/model.json, dan reports/export.md.
"""

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper, numpy_helper
from torch.utils.data import DataLoader

from wastesorter.features import CROP_SIZE, IMAGENET_MEAN, IMAGENET_STD, RESIZE_SIZE, eval_transform
from wastesorter.manifest import MANIFEST_PATH, read_manifest
from wastesorter.model import CHECKPOINT_PATH, load_checkpoint
from wastesorter.report import markdown_table
from wastesorter.train import LabeledImages

IMAGE_ROOT = Path("data/raw/original")
WORK_DIR = Path("runs/export")
WEB_MODEL_DIR = Path("web/model")
MODEL_FILE = "waste.onnx"
REPORT_PATH = Path("reports/export.md")
MODEL_METRICS_PATH = Path("reports/metrics/model.json")

INPUT_NAME = "input"
OUTPUT_NAME = "logits"
OPSET_VERSION = 17
WEIGHT_OPS = ("Conv", "Gemm")
INT8_MAX = 127
# Kriteria docs/PLAN.md: kuantisasi boleh menurunkan akurasi paling banyak 1 poin persentase
MAX_QUANTIZATION_DROP = 0.01
MAX_MODEL_BYTES = 5 * 1024 * 1024
BATCH_SIZE = 32


def export_onnx(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.eval(),
        torch.zeros(1, 3, CROP_SIZE, CROP_SIZE),
        str(path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=OPSET_VERSION,
        dynamo=False,
    )


def _quantize_per_channel(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """int8 simetris per kanal keluaran: weights ~= quantized * scale[kanal]."""
    per_channel_max = np.abs(weights).reshape(weights.shape[0], -1).max(axis=1)
    scale = np.where(per_channel_max > 0, per_channel_max / INT8_MAX, 1.0).astype(np.float32)
    shaped_scale = scale.reshape(-1, *([1] * (weights.ndim - 1)))
    quantized = np.clip(np.round(weights / shaped_scale), -INT8_MAX, INT8_MAX).astype(np.int8)
    return quantized, scale


def quantize_weights(float_path: Path, quantized_path: Path) -> None:
    """Simpan bobot Conv/Gemm sebagai int8 + DequantizeLinear; perhitungan tetap float32.

    Kuantisasi statis penuh (aktivasi ikut int8) dicoba lebih dulu dan ditolak:
    pada MobileNetV3 prediksinya hanya sepakat dengan model asli di <= 42% gambar.
    """
    model = onnx.load(str(float_path))
    graph = model.graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    dequantize_nodes, replaced = [], {}
    for node in graph.node:
        if node.op_type not in WEIGHT_OPS or node.input[1] not in initializers:
            continue
        name = node.input[1]
        if name not in replaced:
            quantized, scale = _quantize_per_channel(numpy_helper.to_array(initializers[name]))
            graph.initializer.remove(initializers[name])
            graph.initializer.extend(
                [
                    numpy_helper.from_array(quantized, f"{name}_int8"),
                    numpy_helper.from_array(scale, f"{name}_scale"),
                    numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), f"{name}_zero"),
                ]
            )
            dequantize_nodes.append(
                helper.make_node(
                    "DequantizeLinear",
                    [f"{name}_int8", f"{name}_scale", f"{name}_zero"],
                    [f"{name}_float"],
                    axis=0,
                )
            )
            replaced[name] = f"{name}_float"
        node.input[1] = replaced[name]
    original_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(dequantize_nodes + original_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(quantized_path))


def onnx_session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def onnx_logits(session: ort.InferenceSession, batch: np.ndarray) -> np.ndarray:
    return session.run([OUTPUT_NAME], {INPUT_NAME: batch})[0]


@dataclass(frozen=True)
class Comparison:
    images: int
    max_logit_difference: float  # PyTorch vs ONNX float32
    torch_accuracy: float
    float_accuracy: float
    quantized_accuracy: float
    quantized_agreement: float  # bagian gambar dengan prediksi sama seperti PyTorch


@torch.no_grad()
def compare(
    model: torch.nn.Module, float_path: Path, quantized_path: Path, loader: DataLoader
) -> Comparison:
    float_session, quantized_session = onnx_session(float_path), onnx_session(quantized_path)
    targets, torch_pred, float_pred, quantized_pred, max_difference = [], [], [], [], 0.0
    for images, batch_targets in loader:
        reference = model(images).numpy()
        float_out = onnx_logits(float_session, images.numpy())
        quantized_out = onnx_logits(quantized_session, images.numpy())
        max_difference = max(max_difference, float(np.abs(reference - float_out).max()))
        targets.append(batch_targets.numpy())
        torch_pred.append(reference.argmax(axis=1))
        float_pred.append(float_out.argmax(axis=1))
        quantized_pred.append(quantized_out.argmax(axis=1))
    targets, torch_pred = np.concatenate(targets), np.concatenate(torch_pred)
    float_pred, quantized_pred = np.concatenate(float_pred), np.concatenate(quantized_pred)
    return Comparison(
        images=len(targets),
        max_logit_difference=max_difference,
        torch_accuracy=float((torch_pred == targets).mean()),
        float_accuracy=float((float_pred == targets).mean()),
        quantized_accuracy=float((quantized_pred == targets).mean()),
        quantized_agreement=float((quantized_pred == torch_pred).mean()),
    )


def _loader(rows: list[dict], labels: list[str], image_root: Path) -> DataLoader:
    index_of = {label: index for index, label in enumerate(labels)}
    dataset = LabeledImages(
        [image_root / row["path"] for row in rows],
        [index_of[row["label"]] for row in rows],
        eval_transform(),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE)


def write_model_config(
    path: Path, labels: list[str], threshold: float, metrics: dict | None = None
) -> None:
    """Semua yang dibutuhkan pemakai model untuk preprocessing dan menafsirkan keluaran."""
    config = {
        "modelFile": MODEL_FILE,
        "labels": labels,
        "threshold": threshold,
        "inputSize": CROP_SIZE,
        "cropFraction": CROP_SIZE / RESIZE_SIZE,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }
    if metrics:
        config["metrics"] = metrics
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _megabytes(path: Path) -> str:
    return f"{path.stat().st_size / 1024 / 1024:.2f} MB"


def write_report(path: Path, result: Comparison, float_path: Path, quantized_path: Path, shipped: Path) -> None:
    drop = result.torch_accuracy - result.quantized_accuracy
    size_ok = shipped.stat().st_size <= MAX_MODEL_BYTES
    sections = [
        "# Laporan Ekspor (Tahap 4)",
        "Dihasilkan oleh `python -m wastesorter.export`. Jangan diedit manual.",
        f"Dibandingkan pada {result.images} gambar validasi, dengan preprocessing yang sama.",
        markdown_table(
            ["Model", "Ukuran", "Akurasi validasi"],
            [
                ["PyTorch", "-", f"{result.torch_accuracy:.4f}"],
                ["ONNX float32", _megabytes(float_path), f"{result.float_accuracy:.4f}"],
                ["ONNX bobot int8", _megabytes(quantized_path), f"{result.quantized_accuracy:.4f}"],
            ],
        ),
        markdown_table(
            ["Pemeriksaan", "Nilai"],
            [
                ["Selisih logit maksimum PyTorch vs ONNX float32", f"{result.max_logit_difference:.2e}"],
                ["Prediksi int8 sama dengan PyTorch", f"{result.quantized_agreement:.1%}"],
                [
                    "Penurunan akurasi akibat kuantisasi",
                    f"{drop * 100:.2f} poin (batas {MAX_QUANTIZATION_DROP * 100:.0f}: "
                    f"{'lolos' if drop <= MAX_QUANTIZATION_DROP else 'TIDAK lolos'})",
                ],
                [
                    "Model yang dipakai app",
                    f"{'int8' if shipped == quantized_path else 'float32'}, {_megabytes(shipped)} "
                    f"(target ≤ 5 MB: {'tercapai' if size_ok else 'TIDAK tercapai'})",
                ],
            ],
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    if not MODEL_METRICS_PATH.is_file():
        raise SystemExit(f"{MODEL_METRICS_PATH} belum ada. Jalankan dulu `python -m wastesorter.evaluate`.")
    evaluation = json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    model, labels, _ = load_checkpoint(args.checkpoint)
    rows = read_manifest(args.manifest)
    val_rows = [row for row in rows if row["split"] == "val"]

    float_path, quantized_path = WORK_DIR / "float32.onnx", WORK_DIR / "int8.onnx"
    export_onnx(model, float_path)
    quantize_weights(float_path, quantized_path)

    result = compare(model, float_path, quantized_path, _loader(val_rows, labels, args.image_root))
    drop = result.torch_accuracy - result.quantized_accuracy
    shipped = quantized_path if drop <= MAX_QUANTIZATION_DROP else float_path
    WEB_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(shipped, WEB_MODEL_DIR / MODEL_FILE)
    write_model_config(
        WEB_MODEL_DIR / "model.json",
        labels,
        evaluation["threshold"],
        {"testAccuracy": evaluation["test_accuracy"], "testMacroF1": evaluation["test_macro_f1"]},
    )
    write_report(REPORT_PATH, result, float_path, quantized_path, shipped)
    print(
        f"Dipakai: {shipped.name} ({_megabytes(shipped)}), akurasi validasi "
        f"{result.quantized_accuracy:.4f} int8 vs {result.torch_accuracy:.4f} PyTorch -> {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
