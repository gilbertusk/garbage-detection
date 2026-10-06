# Waste Sorter — Rencana Proyek

Dibuat: 2026-10-06

## 1. Masalah dan tujuan

Orang sering ragu sampah tertentu masuk tempat sampah yang mana. App ini
menjawabnya dari kamera HP: pengguna mengarahkan kamera ke satu benda, app
menampilkan jenisnya (mis. "Plastik · yakin 92%") dan tempat sampah yang tepat
(mis. "→ Anorganik / daur ulang"). Semua berjalan di browser; gambar tidak
pernah keluar dari perangkat.

Tujuan kedua: menjadi proyek portofolio yang angkanya bisa dipercaya dan
diulang, termasuk hasil pada foto dunia nyata.

## 2. Lingkup

Dikerjakan:

- Klasifikasi gambar tunggal, satu benda di tengah bingkai, 10 kelas.
- Jawaban "tidak yakin" bila kepercayaan model di bawah ambang.
- Web app kamera statis, di-deploy ke GitHub Pages.

Sengaja tidak dikerjakan:

- Deteksi banyak benda sekaligus (object detection).
- App native Android/iOS.
- Server, akun pengguna, atau penyimpanan foto.
- Training ulang dengan foto dunia nyata (foto itu hanya untuk uji).

## 3. Kriteria sukses

Angka di bawah adalah usulan awal. Ditetapkan sebelum training dan tidak
diubah setelah hasil test keluar; kalau meleset, dilaporkan apa adanya.

| Kriteria | Target |
|---|---|
| Macro-F1 pada test set | ≥ 0,90 |
| F1 kelas terlemah pada test set | ≥ 0,80 |
| Kenaikan macro-F1 atas baseline | dilaporkan |
| Akurasi pada ±100 foto HP sendiri | dilaporkan, tanpa target |
| Ukuran model ONNX terkuantisasi | ≤ 5 MB |
| Penurunan akurasi akibat kuantisasi | ≤ 1 poin persentase |
| Waktu inferensi di HP kelas menengah | ≤ 300 ms per gambar |
| Request jaringan yang membawa gambar | 0 (diuji e2e) |

## 4. Dataset

[Garbage Dataset v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2),
lisensi MIT, 12.259 gambar di folder `original/` (halaman dataset menulis total
13.348, tetapi jumlah per kelasnya sendiri berjumlah 12.259, sama dengan isi
file yang diunduh):

| Kelas | Jumlah | Tempat sampah |
|---|---|---|
| clothes | 1.892 | 🟣 Donasi / tekstil |
| glass | 1.736 | 🟡 Anorganik / daur ulang |
| plastic | 1.597 | 🟡 Anorganik / daur ulang |
| shoes | 1.449 | 🟣 Donasi / tekstil |
| cardboard | 1.411 | 🔵 Kertas |
| paper | 1.336 | 🔵 Kertas |
| metal | 930 | 🟡 Anorganik / daur ulang |
| battery | 756 | 🔴 B3 (berbahaya) |
| biological | 699 | 🟢 Organik |
| trash | 453 | ⚫ Residu |

Pemetaan tempat sampah adalah contoh dan perlu disesuaikan dengan aturan
daerah. Jumlah per kelas sudah diverifikasi dari file di tahap 0
(`reports/data.md`). Unduhan juga berisi `standardized_256/` dan
`standardized_384/` (gambar yang sama, sudah di-resize); proyek ini memakai
`original/`.

Set uji dunia nyata: ±100 foto HP sendiri (±10 per kelas) ditambah beberapa
foto non-sampah, disimpan di `data/real/`.

## 5. Tahapan

Setiap tahap selesai bila hasilnya ada di repo dan tesnya lulus.

### Tahap 0 — Data

- Validasi file: gambar rusak dibuang, ukuran dan format dicatat.
- Dedup: salinan byte-identik dibuang (sha1); gambar hampir-sama dikelompokkan
  lewat korelasi thumbnail grayscale. dHash dicoba lebih dulu dan ditolak
  karena foto berlatar polos saling terangkai menjadi grup palsu.
- Split 70/15/15 per kelas, satu kelompok duplikat selalu di split yang sama.
- Selesai bila: `data/manifests/split.csv` (path, kelas, split, grup, sha1)
  dan `reports/data.md` (jumlah per kelas, duplikat, contoh gambar) ada, dan
  tes membuktikan tidak ada grup yang melintasi split.

### Tahap 1 — Baseline

- Fitur MobileNetV3 beku + regresi logistik, dievaluasi di validasi.
- Selesai bila: `reports/baseline.md` berisi akurasi dan macro-F1.

### Tahap 2 — Model

- Fine-tune MobileNetV3-Large (ImageNet, 224×224) dengan augmentasi (putar,
  cahaya, crop) dan bobot kelas.
- Pengaturan di `configs/`, seed tetap.
- Model terbaik dan ambang "tidak yakin" dipilih dari validasi saja.
- Test set dipakai sekali di akhir.
- Selesai bila: `reports/model.md` berisi akurasi, macro-F1, F1 per kelas,
  confusion matrix, dan perbandingan dengan baseline.

### Tahap 3 — Uji dunia nyata

- Model dijalankan pada foto HP sendiri dan foto non-sampah.
- Selesai bila: `reports/real_world.md` berisi akurasi, selisih dengan test
  set, kelas yang paling sering tertukar, dan seberapa sering foto non-sampah
  dijawab "tidak yakin".

### Tahap 4 — App

- Ekspor PyTorch → ONNX → kuantisasi, dengan tes paritas prediksi.
- Web app di `web/`: kamera → crop tengah → klasifikasi → tempat sampah +
  3 tebakan teratas. JavaScript tanpa build step, ONNX Runtime Web (WASM).
- Preprocessing JavaScript diuji identik dengan preprocessing Python.
- Selesai bila: demo berjalan di HP dan benchmark ukuran/waktu tercatat.

### Tahap 5 — Rilis

- CI GitHub Actions: lint, pytest, node:test, Playwright e2e.
- Deploy otomatis ke GitHub Pages.
- README: hasil, perbandingan test vs dunia nyata, keterbatasan, cara
  mengulang.
- Selesai bila: CI hijau dan link demo bisa dibuka.

## 6. Risiko

| Risiko | Penanganan |
|---|---|
| Duplikat bocor ke test, akurasi tinggi palsu | Dedup sebelum split, split per grup, diuji |
| Kelas tidak seimbang (trash 453 vs clothes 1.892) | Bobot kelas; macro-F1 sebagai metrik utama |
| Beda domain: foto Kaggle bersih, foto nyata berantakan | Tahap 3 mengukurnya; hasil dilaporkan apa adanya |
| Kelas ambigu (kemasan berlapis, botol berlabel) | Ambang "tidak yakin"; tampilkan 3 tebakan teratas |
| Benda non-sampah dijawab dengan yakin | Diuji khusus di tahap 3; dicatat sebagai keterbatasan |
| Preprocessing Python dan JavaScript berbeda | Tes paritas pada gambar yang sama |

## 7. Keputusan teknis

- **MobileNetV3-Large**, bukan EfficientNet-Lite: ekspor ONNX dan kuantisasinya
  paling mulus, ukurannya masuk target.
- **PyTorch** untuk training; CPU cukup untuk tahap 0–1, GPU gratis Kaggle
  Notebook untuk tahap 2.
- **ONNX Runtime Web (WASM)** tanpa build step: bisa di-host statis dan gambar
  tetap di perangkat.
- **Manifest split di-commit, gambar tidak**: split bisa diulang persis tanpa
  menyimpan data di git.
- **Pengujian**: pytest (pipeline), node:test (logika app), Playwright (e2e
  dan cek privasi).

## 8. Struktur repo

```
data/raw/original/   dataset Kaggle (tidak masuk git)
data/real/           foto HP sendiri (tidak masuk git)
data/manifests/      file split (masuk git)
src/wastesorter/     data, dedup, train, evaluate, export
configs/             pengaturan training
tests/               pytest
reports/             laporan tiap tahap
web/                 app + model ONNX + node:test
e2e/                 Playwright
.github/workflows/   CI dan deploy
```
