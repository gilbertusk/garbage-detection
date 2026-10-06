# Pilah: pemilah sampah dengan AI di browser

Landing page tentang pemilahan sampah dengan bagian uji foto: arahkan kamera HP
ke satu sampah, lalu situs menampilkan jenisnya dan tempat sampah yang tepat.
Model berjalan di perangkat pengguna; tidak ada server dan foto tidak pernah
dikirim ke mana pun.

Rencana dan kriteria sukses ditetapkan sebelum training: [docs/PLAN.md](docs/PLAN.md).

## Hasil

| Hal | Nilai | Target |
|---|---|---|
| Macro-F1 pada test set (1.836 gambar) | 0,939 | ≥ 0,90 |
| Akurasi pada test set | 0,944 | - |
| F1 kelas terlemah (trash) | 0,867 | ≥ 0,80 |
| Baseline: fitur beku + regresi logistik (macro-F1 validasi) | 0,912 | - |
| Model fine-tune (macro-F1 validasi) | 0,946 | - |
| Ukuran model di situs (ONNX, bobot int8) | 4,2 MB | ≤ 5 MB |
| Penurunan akurasi akibat kuantisasi | 0,33 poin | ≤ 1 poin |
| Akurasi pada foto HP sendiri | belum diukur | dilaporkan |
| Waktu inferensi di HP | belum diukur | ≤ 300 ms |

Test set dipakai satu kali, setelah model dan ambang dipilih dari validasi.
Rincian: [data](reports/data.md), [baseline](reports/baseline.md),
[model](reports/model.md), [ekspor](reports/export.md).

Di bawah ambang kepercayaan 0,58 situs menjawab "tidak yakin". Pada test set
model menjawab 92,8% gambar, dengan akurasi 97,9% di antara yang dijawab.

## Keterbatasan

- **Belum diuji pada foto dunia nyata.** Angka di atas berasal dari test set
  Kaggle, yang gambarnya mirip data latih. Akurasi pada foto HP sehari-hari
  hampir pasti lebih rendah. Skrip ujinya sudah ada
  (`python -m wastesorter.real_world`); fotonya belum dikumpulkan.
- **Kebocoran sesi foto.** Dedup menangkap salinan dan hasil resize, tetapi
  foto berbeda dari sesi pemotretan yang sama (latar identik) tetap tersebar
  di train dan test, sehingga angka test sedikit terlalu optimis.
- **Benda bukan sampah.** Model hanya mengenal sepuluh kelas dan bisa menjawab
  dengan yakin untuk tangan, meja, atau wajah.
- **Yang paling sering tertukar:** kaca dengan plastik, dan kertas dengan kardus.
- **Resize berbeda antara browser dan Python.** Normalisasi piksel diuji
  identik, tetapi algoritme resize kanvas browser tidak sama dengan Pillow.
  Tes e2e memeriksa tiga foto memberi jawaban yang sama, bukan seluruh test set.
- **Training singkat:** 5 epoch di CPU. Epoch terakhir masih yang terbaik, jadi
  training lebih lama kemungkinan masih menaikkan angka.
- **Pemetaan tempat sampah** bersifat umum; aturan tiap daerah berbeda.

## Dataset

[Garbage Dataset v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2),
lisensi MIT, 10 kelas. Folder `original/` berisi 12.259 gambar (halaman dataset
menulis 13.348, tetapi jumlah per kelasnya berjumlah 12.259). Unduh dari Kaggle
dan ekstrak sehingga ada `data/raw/original/<kelas>/`; folder itu tidak masuk git.

## Mengulang hasil

```
pip install -e ".[dev]"
python -m wastesorter.prepare      # validasi, dedup, split -> reports/data.md
python -m wastesorter.baseline     # baseline -> reports/baseline.md
python -m wastesorter.train        # fine-tune (±75 menit di CPU 4 inti)
python -m wastesorter.evaluate     # test set -> reports/model.md
python -m wastesorter.export       # ONNX + bobot int8 -> web/model/, reports/export.md
```

Split tersimpan di `data/manifests/split.csv` dan pengaturan training di
`configs/train.json`, keduanya dengan seed tetap.

## Situs

`web/` adalah situs statis tanpa build step dan tanpa CDN: ONNX Runtime Web
dan font disalin ke `web/vendor/` oleh `npm run vendor`.

```
npm install
npm run vendor
npm run serve     # http://localhost:8000
```

## Pengujian

```
ruff check src tests
python -m pytest          # pipeline data, training, ekspor, prediksi
npm test                  # logika situs (node:test)
npx playwright install chromium
npm run e2e               # alur situs di desktop dan HP, termasuk cek privasi
```

Tes e2e memastikan tidak ada request jaringan setelah foto dipilih, dan semua
request halaman hanya GET ke asal yang sama.

## Memakai model di tempat lain

Hasil ekspor ada di `web/model/`: `waste.onnx` (model) dan `model.json` (label,
ambang "tidak yakin", ukuran input, mean/std). Dua berkas itu cukup untuk
menjalankan model di mana pun ONNX Runtime tersedia.

Python, tanpa PyTorch:

```python
from wastesorter.predict import WasteClassifier

classifier = WasteClassifier("web/model")
prediction = classifier.predict("foto.jpg")
print(prediction.is_confident, prediction.top)
```

Atau dari terminal: `python -m wastesorter.predict foto.jpg`.

JavaScript di halaman lain:

```js
import { createClassifier } from "./src/classifier.js";

const classifier = await createClassifier();
const result = await classifier.classify(img, img.naturalWidth, img.naturalHeight);
```

Preprocessing harus sama dengan saat training: sisi terpendek diubah ke 256,
crop tengah 224, lalu normalisasi dengan mean/std di `model.json`.

## Struktur

```
src/wastesorter/   data, dedup, split, baseline, train, evaluate, export, predict
tests/             pytest
configs/           pengaturan training
data/manifests/    split yang di-commit (gambarnya tidak)
reports/           laporan tiap tahap
web/               situs, model ONNX, tes node:test
e2e/               Playwright
.github/workflows/ CI dan deploy GitHub Pages
```
