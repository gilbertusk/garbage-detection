"""Deteksi gambar hampir-sama lewat korelasi thumbnail grayscale.

dHash 64-bit sempat dicoba dan ditolak: pada dataset ini foto berlatar polos
saling berdekatan hash-nya walau bendanya berbeda, sehingga ratusan gambar
tak berhubungan terangkai menjadi satu grup.
"""

import numpy as np
from PIL import Image

SIGNATURE_SIDE = 32
NORM_EPSILON = 1e-6


def thumbnail_signature(image: Image.Image) -> bytes:
    """Thumbnail grayscale SIGNATURE_SIDE x SIGNATURE_SIDE sebagai bytes."""
    gray = image.convert("L").resize(
        (SIGNATURE_SIDE, SIGNATURE_SIDE), Image.Resampling.LANCZOS
    )
    return gray.tobytes()


def _normalize(signatures: list[bytes]) -> np.ndarray:
    """Ubah ke vektor rata-rata nol dan panjang satu, agar dot product = korelasi."""
    pixels = np.frombuffer(b"".join(signatures), dtype=np.uint8)
    vectors = pixels.reshape(len(signatures), -1).astype(np.float32)
    vectors = vectors - vectors.mean(axis=1, keepdims=True)
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + NORM_EPSILON)


def correlation(first: bytes, second: bytes) -> float:
    vectors = _normalize([first, second])
    return float(vectors[0] @ vectors[1])


def _find(parents: list[int], index: int) -> int:
    root = index
    while parents[root] != root:
        root = parents[root]
    while parents[index] != root:
        parents[index], index = root, parents[index]
    return root


def group_near_duplicates(
    signatures: list[bytes], min_correlation: float, chunk_size: int = 2048
) -> np.ndarray:
    """Kelompokkan gambar yang korelasinya >= min_correlation (transitif).

    Mengembalikan id grup per elemen; id-nya adalah indeks terkecil di grup itu.
    """
    count = len(signatures)
    vectors = _normalize(signatures)

    parents = list(range(count))
    for start in range(0, count, chunk_size):
        correlations = vectors[start : start + chunk_size] @ vectors.T
        rows, cols = np.nonzero(correlations >= min_correlation)
        for row, col in zip((rows + start).tolist(), cols.tolist()):
            if row >= col:
                continue
            root_a, root_b = _find(parents, row), _find(parents, col)
            if root_a != root_b:
                parents[max(root_a, root_b)] = min(root_a, root_b)

    return np.array([_find(parents, index) for index in range(count)], dtype=np.int64)
