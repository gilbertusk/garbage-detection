# Laporan Baseline (Tahap 1)

Dihasilkan oleh `python -m wastesorter.baseline`. Jangan diedit manual.

Fitur 960-dimensi dari MobileNetV3-Large beku (ImageNet), lalu standardisasi dan regresi logistik (C=1.0, bobot kelas seimbang). Dilatih pada 8580 gambar train, dinilai pada 1843 gambar validasi. Test set tidak disentuh.

## Hasil di validasi

| Model | Akurasi | Macro-F1 |
|---|---|---|
| Selalu tebak kelas terbanyak | 0.154 | 0.027 |
| **MobileNetV3 beku + regresi logistik** | **0.919** | **0.912** |

## F1 per kelas

| Kelas | F1 | Jumlah di validasi |
|---|---|---|
| battery | 0.961 | 114 |
| biological | 0.962 | 105 |
| cardboard | 0.925 | 212 |
| clothes | 0.993 | 284 |
| glass | 0.898 | 261 |
| metal | 0.835 | 140 |
| paper | 0.886 | 201 |
| plastic | 0.864 | 240 |
| shoes | 0.970 | 218 |
| trash | 0.830 | 68 |

## Confusion matrix

Baris = label benar, kolom = prediksi.

|  | battery | biological | cardboard | clothes | glass | metal | paper | plastic | shoes | trash |
|---|---|---|---|---|---|---|---|---|---|---|
| battery | 111 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| biological | 0 | 101 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 1 |
| cardboard | 1 | 0 | 198 | 0 | 1 | 2 | 9 | 0 | 0 | 1 |
| clothes | 1 | 0 | 0 | 280 | 0 | 1 | 0 | 1 | 1 | 0 |
| glass | 0 | 0 | 1 | 0 | 233 | 8 | 0 | 15 | 2 | 2 |
| metal | 1 | 0 | 1 | 0 | 9 | 119 | 2 | 7 | 0 | 1 |
| paper | 2 | 2 | 8 | 0 | 0 | 2 | 178 | 5 | 1 | 3 |
| plastic | 0 | 0 | 6 | 0 | 11 | 6 | 7 | 207 | 1 | 2 |
| shoes | 0 | 1 | 0 | 0 | 3 | 0 | 1 | 2 | 210 | 1 |
| trash | 1 | 1 | 1 | 0 | 0 | 4 | 3 | 2 | 0 | 56 |
