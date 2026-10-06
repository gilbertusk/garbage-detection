// Tampilan bagian "Uji foto": kamera/foto -> pengklasifikasi -> kartu hasil.

import { describeClass } from "./bins.js";
import { createClassifier } from "./classifier.js";

const CAMERA_INTERVAL_MS = 400;
const UNSURE_COLOR = "#4b5157";

const elements = {
  camera: document.querySelector("#camera"),
  photo: document.querySelector("#photo"),
  status: document.querySelector("#status"),
  result: document.querySelector("#result"),
  bin: document.querySelector("#bin"),
  guess: document.querySelector("#guess"),
  tip: document.querySelector("#tip"),
  guesses: document.querySelector("#guesses"),
  timing: document.querySelector("#timing"),
  startCamera: document.querySelector("#start-camera"),
  pickPhoto: document.querySelector("#pick-photo"),
  metrics: document.querySelector("#metrics"),
  metricsMissing: document.querySelector("#metrics-missing"),
};

const percent = (value) => `${Math.round(value * 100)}%`;

function guessRow({ label, confidence }) {
  const item = document.createElement("li");
  const name = document.createElement("span");
  name.textContent = describeClass(label).name;
  const bar = document.createElement("span");
  bar.className = "bar";
  const fill = document.createElement("span");
  fill.style.width = percent(confidence);
  bar.append(fill);
  const value = document.createElement("span");
  value.className = "percent";
  value.textContent = percent(confidence);
  item.append(name, bar, value);
  return item;
}

function renderResult({ isConfident, top, milliseconds }) {
  const best = describeClass(top[0].label);
  if (isConfident) {
    elements.bin.textContent = best.bin.name;
    elements.guess.textContent = `${best.name}, yakin ${percent(top[0].confidence)}`;
    elements.tip.textContent = best.tip;
  } else {
    elements.bin.textContent = "Tidak yakin";
    elements.guess.textContent = "Model tidak cukup yakin untuk menjawab.";
    elements.tip.textContent = "Dekatkan kamera, tambah cahaya, atau coba dari sudut lain.";
  }
  elements.result.style.setProperty("--bin", isConfident ? best.bin.color : UNSURE_COLOR);
  elements.result.dataset.confident = String(isConfident);
  elements.result.dataset.label = top[0].label;
  elements.guesses.replaceChildren(...top.map(guessRow));
  elements.timing.textContent = `Dihitung dalam ${Math.round(milliseconds)} ms di perangkat ini.`;
  elements.result.hidden = false;
}

/** Tampilkan angka evaluasi yang ikut diekspor bersama model (kalau ada). */
function renderMetrics(metrics) {
  if (!metrics) {
    return;
  }
  const rows = [
    ["Akurasi pada data uji", percent(metrics.testAccuracy)],
    ["Macro-F1 pada data uji", percent(metrics.testMacroF1)],
  ];
  elements.metrics.replaceChildren(
    ...rows.map(([label, value]) => {
      const row = document.createElement("div");
      const term = document.createElement("dt");
      term.textContent = label;
      const detail = document.createElement("dd");
      detail.textContent = value;
      row.append(term, detail);
      return row;
    }),
  );
  elements.metrics.hidden = false;
  elements.metricsMissing.hidden = true;
}

function showStatus(message) {
  elements.status.textContent = message;
}

function stopCamera() {
  const stream = elements.camera.srcObject;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    elements.camera.srcObject = null;
  }
  elements.camera.hidden = true;
}

async function startCamera(classify) {
  if (!navigator.mediaDevices?.getUserMedia) {
    showStatus("Browser ini tidak mendukung kamera. Pakai tombol Pilih foto.");
    return;
  }
  try {
    elements.camera.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false,
    });
  } catch (error) {
    console.error("Kamera gagal dibuka", error);
    showStatus("Kamera tidak bisa dibuka. Izinkan akses kamera, atau pakai tombol Pilih foto.");
    return;
  }
  elements.photo.hidden = true;
  elements.camera.hidden = false;
  await elements.camera.play();
  showStatus("");

  const stream = elements.camera.srcObject;
  const loop = async () => {
    if (elements.camera.srcObject !== stream) {
      return; // kamera dimatikan atau diganti
    }
    try {
      const { videoWidth, videoHeight } = elements.camera;
      if (videoWidth > 0) {
        renderResult(await classify(elements.camera, videoWidth, videoHeight));
      }
    } catch (error) {
      console.error("Klasifikasi gagal", error);
      showStatus("Pengenalan gagal. Muat ulang halaman.");
      return;
    }
    setTimeout(loop, CAMERA_INTERVAL_MS);
  };
  loop();
}

async function classifyPhoto(classify, file) {
  stopCamera();
  const url = URL.createObjectURL(file);
  try {
    elements.photo.src = url;
    await elements.photo.decode();
    elements.photo.hidden = false;
    showStatus("");
    const { naturalWidth, naturalHeight } = elements.photo;
    renderResult(await classify(elements.photo, naturalWidth, naturalHeight));
  } catch (error) {
    console.error("Foto gagal diproses", error);
    showStatus("Foto ini tidak bisa dibaca. Coba foto lain berformat JPG atau PNG.");
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function main() {
  let classifier;
  try {
    classifier = await createClassifier();
  } catch (error) {
    console.error("Model gagal dimuat", error);
    showStatus("Model gagal dimuat. Periksa koneksi, lalu muat ulang halaman.");
    return;
  }
  renderMetrics(classifier.config.metrics);
  showStatus("Model siap. Nyalakan kamera atau pilih foto.");
  document.body.dataset.ready = "true";

  elements.startCamera.disabled = false;
  elements.pickPhoto.disabled = false;
  elements.startCamera.addEventListener("click", () => startCamera(classifier.classify));
  elements.pickPhoto.addEventListener("change", () => {
    const [file] = elements.pickPhoto.files;
    if (file) {
      classifyPhoto(classifier.classify, file);
    }
  });
}

main();
