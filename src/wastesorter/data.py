"""Pemindaian folder dataset: validasi file, ukuran, sha1, dan thumbnail."""

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from wastesorter.dedup import thumbnail_signature

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DRAFT_SIZE = (64, 64)


@dataclass(frozen=True)
class ImageRecord:
    path: str  # relatif terhadap root, pemisah "/"
    label: str
    width: int
    height: int
    format: str
    sha1: str
    signature: bytes  # thumbnail grayscale untuk deteksi hampir-sama


def _read_record(root: Path, file: Path) -> ImageRecord | None:
    relative = file.relative_to(root).as_posix()
    try:
        content = file.read_bytes()
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
            image_format = image.format or ""
            # draft mempercepat decode JPEG; gambar tetap di-decode sehingga
            # file terpotong/rusak tetap terdeteksi
            image.draft("L", DRAFT_SIZE)
            signature = thumbnail_signature(image)
    except (OSError, ValueError, SyntaxError):
        return None
    return ImageRecord(
        path=relative,
        label=file.parent.name,
        width=width,
        height=height,
        format=image_format,
        sha1=hashlib.sha1(content).hexdigest(),
        signature=signature,
    )


def scan_images(root: Path, workers: int = 8) -> tuple[list[ImageRecord], list[str]]:
    """Baca semua gambar di root/<kelas>/. Mengembalikan (record valid, path rusak)."""
    root = Path(root)
    files = sorted(
        file
        for file in root.glob("*/*")
        if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda file: _read_record(root, file), files))

    records = [record for record in results if record is not None]
    invalid = [
        file.relative_to(root).as_posix()
        for file, record in zip(files, results)
        if record is None
    ]
    return records, invalid
