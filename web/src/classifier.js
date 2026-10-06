// Pengklasifikasi sampah yang bisa dipakai ulang di halaman lain.
//
//   import { createClassifier } from "./src/classifier.js";
//   const classifier = await createClassifier();
//   const result = await classifier.classify(imgElement, img.naturalWidth, img.naturalHeight);
//   // result = { isConfident, top: [{ label, confidence }, ...], milliseconds }
//
// Tidak menyentuh DOM halaman dan tidak mengirim gambar ke mana pun.

import * as ort from "../vendor/ort/ort.wasm.min.mjs";

import { interpret } from "./classify.js";
import { centerCropRect, toTensorData } from "./preprocess.js";

const DEFAULT_MODEL_DIR = new URL("../model/", import.meta.url);
const DEFAULT_ORT_DIR = new URL("../vendor/ort/", import.meta.url);
const RGB_CHANNELS = 3;

/**
 * @param {{modelDir?: URL, ortDir?: URL}} [options] lokasi model.json + berkas ONNX,
 *   dan lokasi berkas WASM ONNX Runtime
 * @returns {Promise<{config: object, classify: Function}>}
 */
export async function createClassifier({
  modelDir = DEFAULT_MODEL_DIR,
  ortDir = DEFAULT_ORT_DIR,
} = {}) {
  const response = await fetch(new URL("model.json", modelDir));
  if (!response.ok) {
    throw new Error(`model.json tidak bisa dimuat (HTTP ${response.status})`);
  }
  const config = await response.json();

  // Hosting statis seperti GitHub Pages tidak mengirim header untuk
  // SharedArrayBuffer, jadi satu thread
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = ortDir.href;
  const session = await ort.InferenceSession.create(new URL(config.modelFile, modelDir).href, {
    executionProviders: ["wasm"],
  });

  const side = config.inputSize;
  const canvas = document.createElement("canvas");
  canvas.width = side;
  canvas.height = side;
  const context = canvas.getContext("2d", { willReadFrequently: true });

  /** source: <video>, <img>, atau <canvas>; width/height: ukuran asli sumbernya */
  async function classify(source, width, height) {
    const { sx, sy, size } = centerCropRect(width, height, config.cropFraction);
    context.drawImage(source, sx, sy, size, size, 0, 0, side, side);
    const { data } = context.getImageData(0, 0, side, side);
    const input = new ort.Tensor("float32", toTensorData(data, config.mean, config.std), [
      1,
      RGB_CHANNELS,
      side,
      side,
    ]);

    const started = performance.now();
    const output = await session.run({ [session.inputNames[0]]: input });
    const milliseconds = performance.now() - started;

    const logits = output[session.outputNames[0]].data;
    return { ...interpret(logits, config.labels, config.threshold), milliseconds };
  }

  return { config, classify };
}
