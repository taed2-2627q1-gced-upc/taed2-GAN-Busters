"""
Clean and preprocess the (intentionally corrupted) CIFAKE dataset, producing
the final, canonical dataset used for modeling.

Cleaning steps:
    - Detect and discard unreadable/corrupted files
    - Detect and discard duplicate files (by content hash)
    - Denoise images (median filter, reduces Gaussian noise and occlusion
      artifacts)
    - Standardize color mode (ensure 3-channel RGB even for images that were
      corrupted into grayscale)
    - Resize every image to a consistent shape

Usage:
    python -m gan_busters.features
"""

import hashlib
from pathlib import Path

from loguru import logger
from PIL import Image, ImageFilter, UnidentifiedImageError
from tqdm import tqdm
import typer

from gan_busters.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

IMAGE_SIZE = (32, 32)  # matches the original CIFAKE/CIFAR-10 resolution


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def clean_image(img: Image.Image) -> Image.Image:
    """Apply denoising, standardize mode and size."""
    img = img.convert("RGB")
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.resize(IMAGE_SIZE)
    return img


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_dir: Path = INTERIM_DATA_DIR / "cifake_corrupted",
    output_dir: Path = PROCESSED_DATA_DIR / "cifake_clean",
    # ----------------------------------------------
):
    image_paths = [
        p for p in input_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    logger.info(f"Found {len(image_paths)} candidate files in {input_dir}")

    seen_hashes: set[str] = set()
    n_unreadable = 0
    n_duplicate = 0
    n_cleaned = 0

    for src in tqdm(image_paths, total=len(image_paths), desc="Cleaning dataset"):
        rel = src.relative_to(input_dir)
        dst = output_dir / rel

        # 1. Drop unreadable/corrupted files
        try:
            img = Image.open(src)
            img.verify()  # checks integrity without fully decoding
            img = Image.open(src)  # reopen, verify() closes the file handle
        except (UnidentifiedImageError, OSError):
            n_unreadable += 1
            continue

        # 2. Drop duplicates (by content hash)
        h = file_hash(src)
        if h in seen_hashes:
            n_duplicate += 1
            continue
        seen_hashes.add(h)

        # 3. Denoise, standardize mode/size, and save
        cleaned = clean_image(img)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cleaned.save(dst)
        n_cleaned += 1

    logger.success(
        f"Cleaning complete: {n_cleaned} images cleaned, "
        f"{n_unreadable} unreadable files discarded, "
        f"{n_duplicate} duplicates discarded. Output at {output_dir}"
    )


if __name__ == "__main__":
    app()