# Laporan Model (Tahap 2)

Dihasilkan oleh `python -m wastesorter.evaluate`. Jangan diedit manual.

MobileNetV3-Large fine-tune, checkpoint epoch 5 (macro-F1 validasi terbaik). Model dan ambang dipilih dari validasi saja.

## Hasil di test set

| Metrik | Nilai |
|---|---|
| Jumlah gambar | 1836 |
| Akurasi | 0.944 |
| Macro-F1 | 0.939 (target ≥ 0.90: tercapai) |
| F1 kelas terlemah (trash) | 0.867 (target ≥ 0.80: tercapai) |

## Perbandingan dengan baseline (validasi)

| Model | Akurasi | Macro-F1 |
|---|---|---|
| Baseline (fitur beku + regresi logistik) | 0.919 | 0.912 |
| MobileNetV3 fine-tune | 0.951 | 0.946 |

## F1 per kelas (test)

| Kelas | F1 | Jumlah |
|---|---|---|
| battery | 0.978 | 113 |
| biological | 0.958 | 105 |
| cardboard | 0.936 | 211 |
| clothes | 0.991 | 284 |
| glass | 0.940 | 260 |
| metal | 0.915 | 139 |
| paper | 0.909 | 200 |
| plastic | 0.916 | 239 |
| shoes | 0.977 | 217 |
| trash | 0.867 | 68 |

## Confusion matrix (test)

Baris = label benar, kolom = prediksi.

|  | battery | biological | cardboard | clothes | glass | metal | paper | plastic | shoes | trash |
|---|---|---|---|---|---|---|---|---|---|---|
| battery | 110 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 |
| biological | 0 | 102 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 |
| cardboard | 1 | 1 | 197 | 0 | 0 | 3 | 6 | 1 | 0 | 2 |
| clothes | 0 | 0 | 0 | 282 | 0 | 1 | 1 | 0 | 0 | 0 |
| glass | 1 | 1 | 0 | 0 | 243 | 4 | 0 | 10 | 1 | 0 |
| metal | 0 | 0 | 1 | 0 | 0 | 135 | 0 | 2 | 0 | 1 |
| paper | 0 | 4 | 11 | 1 | 1 | 4 | 174 | 2 | 0 | 3 |
| plastic | 0 | 0 | 1 | 0 | 11 | 4 | 0 | 217 | 1 | 5 |
| shoes | 0 | 0 | 0 | 2 | 2 | 1 | 0 | 0 | 211 | 1 |
| trash | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 2 | 1 | 62 |

## Ambang "tidak yakin"

Ambang = nilai terendah di validasi yang membuat akurasi jawaban ≥ 98%. Terpilih: **0.58**. Di bawah ambang, app menjawab "tidak yakin".

Validasi:

| Ambang | Dijawab | Akurasi jawaban |
|---|---|---|
| 0.00 | 100.0% | 0.951 |
| 0.50 | 95.4% | 0.972 |
| **0.58 (terpilih)** | 92.5% | 0.981 |
| 0.60 | 91.8% | 0.983 |
| 0.70 | 87.0% | 0.990 |
| 0.80 | 71.6% | 0.995 |
| 0.90 | 29.0% | 0.994 |

Test (ambang tidak diubah):

| Ambang | Dijawab | Akurasi jawaban |
|---|---|---|
| 0.00 | 100.0% | 0.944 |
| 0.50 | 95.3% | 0.969 |
| **0.58 (terpilih)** | 92.8% | 0.979 |
| 0.60 | 92.0% | 0.980 |
| 0.70 | 87.4% | 0.988 |
| 0.80 | 72.7% | 0.993 |
| 0.90 | 30.2% | 0.993 |
