"""
Generate a reproducible robustness evaluation subset from the validated CIFAKE
dataset.

Sampling:
    - Use data/processed/accepted_records.csv as the source for sampling.
    - Filter to images belonging to the test split only.
    - Use a fixed random seed for reproducibility.
    - Select 1,000 source images:
        - 500 REAL
        - 500 FAKE
    - Within each label, sample exactly 50 images from each of the
      10 CIFAR-10 semantic subclasses.
    - Use relative_path from metadata.csv to locate the selected images
      in data/processed.
    - Training images must not be used for robustness evaluation.

For each selected source image, generate all controlled variants below.

Corruption categories:

1. Unreadable file
    - Deliberately truncate the encoded image file.
    - Used to test input validation and fault tolerance.
    - Expected behavior: the pipeline/API rejects the file safely.

2. Mild Gaussian noise
    - Add low-intensity Gaussian noise.
    - Keep the perturbation mild enough that image content remains
      clearly recognizable.
    - Used to evaluate prediction stability under small pixel-level noise.

3. Small crop + resize
    - Remove exactly 1 pixel from each side of the 32x32 image,
      producing a centered 30x30 crop.
    - Resize the crop back to 32x32 using bilinear interpolation.
    - Do not add padding or artificial pixels.
    - Preserve the original color mode.
    - Used to evaluate robustness to small framing/zoom changes.

4. Horizontal flip
    - Mirror the image horizontally.
    - Used to evaluate whether predictions are unnecessarily sensitive
      to horizontal orientation.

Output:
    data/robustness/cifake/
        original/
        unreadable/
        gaussian_noise/
        crop/
        horizontal_flip/
        metadata.csv

Robustness metadata:
    - source image path
    - generated image path
    - REAL/FAKE label
    - semantic subclass
    - corruption type
    - corruption parameters
    - random seed

The validated dataset in data/processed must never be modified.
"""



# """
# Introduce synthetic corruptions into the clean CIFAKE dataset, to simulate a
# "dirty" real-world dataset that then needs preprocessing/cleaning.

# Corruptions applied to a random subset of images:
#     - Gaussian noise
#     - Black occlusion patches (simulating sensor/transmission artifacts)
#     - Grayscale conversion (simulating loss of color channel information)
#     - A few duplicate files (simulating collection errors)
#     - A few unreadable/corrupted files (simulating file corruption)

# Usage:
#     python -m gan_busters.corrupt_data
# """

# import random
# import shutil
# from pathlib import Path

# from loguru import logger
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import typer

# from gan_busters.config import INTERIM_DATA_DIR, RAW_DATA_DIR

# app = typer.Typer()

# random.seed(42)
# np.random.seed(42)

# # Fraction of images affected by each corruption type
# NOISE_FRACTION = 0.15
# OCCLUSION_FRACTION = 0.10
# GRAYSCALE_FRACTION = 0.10
# DUPLICATE_FRACTION = 0.02
# UNREADABLE_FRACTION = 0.01


# def add_gaussian_noise(img: Image.Image, sigma: float = 25.0) -> Image.Image:
#     arr = np.array(img).astype(np.float32)
#     noise = np.random.normal(0, sigma, arr.shape)
#     noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
#     return Image.fromarray(noisy)


# def add_occlusion(img: Image.Image, patch_size: int = 8) -> Image.Image:
#     arr = np.array(img).copy()
#     h, w = arr.shape[:2]
#     py = random.randint(0, max(h - patch_size, 0))
#     px = random.randint(0, max(w - patch_size, 0))
#     arr[py : py + patch_size, px : px + patch_size] = 0
#     return Image.fromarray(arr)


# def to_grayscale_rgb(img: Image.Image) -> Image.Image:
#     return img.convert("L").convert("RGB")


# def corrupt_image_file(src_path: Path, dst_path: Path) -> None:
#     """Apply at most one random corruption to a single image and save it."""
#     img = Image.open(src_path).convert("RGB")

#     roll = random.random()
#     cumulative = 0.0
#     for fraction, fn in [
#         (NOISE_FRACTION, add_gaussian_noise),
#         (OCCLUSION_FRACTION, add_occlusion),
#         (GRAYSCALE_FRACTION, to_grayscale_rgb),
#     ]:
#         cumulative += fraction
#         if roll < cumulative:
#             img = fn(img)
#             break

#     dst_path.parent.mkdir(parents=True, exist_ok=True)
#     img.save(dst_path)


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_dir: Path = RAW_DATA_DIR / "cifake",
#     output_dir: Path = INTERIM_DATA_DIR / "cifake_corrupted",
#     # ----------------------------------------------
# ):
#     image_paths = [
#         p for p in input_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
#     ]
#     logger.info(f"Found {len(image_paths)} images in {input_dir}")

#     n_unreadable = int(len(image_paths) * UNREADABLE_FRACTION)
#     n_duplicate = int(len(image_paths) * DUPLICATE_FRACTION)
#     unreadable_targets = set(random.sample(image_paths, n_unreadable))
#     duplicate_sources = random.sample(image_paths, n_duplicate)

#     for src in tqdm(image_paths, total=len(image_paths), desc="Corrupting dataset"):
#         rel = src.relative_to(input_dir)
#         dst = output_dir / rel

#         if src in unreadable_targets:
#             # Write a few corrupted bytes instead of a valid image
#             dst.parent.mkdir(parents=True, exist_ok=True)
#             dst.write_bytes(b"not a valid image file")
#             continue

#         corrupt_image_file(src, dst)

#     # Add duplicate files with a "_dup" suffix, copied from the (already
#     # corrupted-or-not) output, to simulate accidental re-collection
#     for src in duplicate_sources:
#         rel = src.relative_to(input_dir)
#         original_out = output_dir / rel
#         if not original_out.exists():
#             continue
#         dup_out = original_out.with_stem(original_out.stem + "_dup")
#         shutil.copy2(original_out, dup_out)

#     logger.success(
#         f"Corruption complete: {n_unreadable} unreadable files, "
#         f"{n_duplicate} duplicates, and noise/occlusion/grayscale applied "
#         f"to ~{int((NOISE_FRACTION + OCCLUSION_FRACTION + GRAYSCALE_FRACTION) * 100)}% "
#         f"of images. Output at {output_dir}"
#     )


# if __name__ == "__main__":
#     app()