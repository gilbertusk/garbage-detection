# Laporan Ekspor (Tahap 4)

Dihasilkan oleh `python -m wastesorter.export`. Jangan diedit manual.

Dibandingkan pada 1843 gambar validasi, dengan preprocessing yang sama.

| Model | Ukuran | Akurasi validasi |
|---|---|---|
| PyTorch | - | 0.9506 |
| ONNX float32 | 16.07 MB | 0.9506 |
| ONNX bobot int8 | 4.20 MB | 0.9474 |

| Pemeriksaan | Nilai |
|---|---|
| Selisih logit maksimum PyTorch vs ONNX float32 | 3.67e-05 |
| Prediksi int8 sama dengan PyTorch | 98.9% |
| Penurunan akurasi akibat kuantisasi | 0.33 poin (batas 1: lolos) |
| Model yang dipakai app | int8, 4.20 MB (target ≤ 5 MB: tercapai) |
