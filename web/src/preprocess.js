// Preprocessing gambar untuk model. Harus identik dengan eval_transform() di
// src/wastesorter/features.py; paritasnya diuji lewat web/tests/fixtures/preprocess.json.

const RGBA_CHANNELS = 4;
const RGB_CHANNELS = 3;
const MAX_PIXEL = 255;

/**
 * Kotak tengah yang dilihat model: sisi = fraction x sisi terpendek.
 * Setara dengan Resize(256) + CenterCrop(224) di Python (fraction = 224/256).
 */
export function centerCropRect(width, height, fraction) {
  if (!(width > 0 && height > 0)) {
    throw new RangeError(`Ukuran gambar tidak valid: ${width}x${height}`);
  }
  const size = Math.round(Math.min(width, height) * fraction);
  return {
    sx: Math.round((width - size) / 2),
    sy: Math.round((height - size) / 2),
    size,
  };
}

/**
 * Piksel RGBA (urut baris, 0-255) -> Float32Array ternormalisasi dengan susunan
 * CHW (semua R, lalu semua G, lalu semua B), seperti ToTensor + Normalize.
 */
export function toTensorData(rgba, mean, std) {
  const pixelCount = rgba.length / RGBA_CHANNELS;
  if (!Number.isInteger(pixelCount)) {
    throw new RangeError("Panjang data RGBA harus kelipatan 4");
  }
  const data = new Float32Array(pixelCount * RGB_CHANNELS);
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    for (let channel = 0; channel < RGB_CHANNELS; channel += 1) {
      const value = rgba[pixel * RGBA_CHANNELS + channel] / MAX_PIXEL;
      data[channel * pixelCount + pixel] = (value - mean[channel]) / std[channel];
    }
  }
  return data;
}
