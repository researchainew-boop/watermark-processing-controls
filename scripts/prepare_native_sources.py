"""Copy native-resolution sources into the study folders.

Real photos are already in real_coco. This script fills a new synthetic
folder with Stable Diffusion (DiffusionDB), DALL-E 3, and held-out Midjourney.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

RAW = Path(r"data/raw")
DOWNLOADS = RAW / "downloads"
SYNTH = RAW / "synthetic_native"
MIN_SIDE = 256


def _save(image: Image.Image, path: Path) -> bool:
    if min(image.size) < MIN_SIDE:
        return False
    image = image.convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=95)
    return True


def _take_zip_images(zip_path: Path, limit: int, stem: str, generator: str, start: int) -> int:
    written = 0
    with zipfile.ZipFile(zip_path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            and "thumb" not in name.lower()
            and not name.endswith("/")
        ]
        for name in names:
            if written >= limit:
                break
            with archive.open(name) as handle:
                image = Image.open(io.BytesIO(handle.read()))
                image.load()
            target = SYNTH / f"{stem}_{start + written:05d}__{generator}.jpg"
            if _save(image, target):
                written += 1
    return written


def _image_from_cell(cell) -> Image.Image | None:
    if cell is None:
        return None
    if isinstance(cell, dict):
        payload = cell.get("bytes")
        if payload:
            return Image.open(io.BytesIO(payload))
        return None
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell))
    return None


def _take_parquet(paths: list[Path], limit: int) -> int:
    written = 0
    for path in paths:
        table = pq.read_table(path)
        frame = table.to_pandas()
        image_column = next(
            (name for name in ("image", "img", "jpg") if name in frame.columns),
            None,
        )
        if image_column is None:
            raise SystemExit(f"No image column in {path.name}: {list(frame.columns)}")
        labels = frame["label"] if "label" in frame.columns else None
        for index, cell in enumerate(frame[image_column]):
            if written >= limit:
                return written
            if labels is not None and int(labels.iloc[index]) != 0:
                continue
            image = _image_from_cell(cell)
            if image is None:
                continue
            target = SYNTH / f"midjourney_{written:05d}__midjourney.jpg"
            if _save(image, target):
                written += 1
    return written


def main() -> None:
    SYNTH.mkdir(parents=True, exist_ok=True)
    for old in SYNTH.glob("*"):
        if old.is_file():
            old.unlink()
    sd = _take_zip_images(DOWNLOADS / "part-000001.zip", 1000, "sd", "stable-diffusion", 0)
    sd += _take_zip_images(DOWNLOADS / "part-000002.zip", 1000, "sd", "stable-diffusion", sd)
    dalle = _take_zip_images(DOWNLOADS / "Dalle3-dataset.zip", 1000, "dalle", "dalle-3", 0)
    midjourney = _take_parquet(
        [
            DOWNLOADS / "mj-000.parquet",
            DOWNLOADS / "mj-001.parquet",
            DOWNLOADS / "mj-002.parquet",
        ],
        1000,
    )
    print(f"stable-diffusion {sd}")
    print(f"dalle-3 {dalle}")
    print(f"midjourney {midjourney}")
    print(f"synthetic total {sd + dalle + midjourney}")


if __name__ == "__main__":
    main()
