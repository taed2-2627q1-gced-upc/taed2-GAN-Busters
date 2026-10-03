"""
Inspect the raw image dataset and generate a file-level inventory for
downstream data integrity checks.

For each discovered file, this stage:
    - extracts the predefined train/test split when available;
    - extracts the REAL/FAKE target label from the directory structure;
    - verifies that the file can be read and decoded as an image;
    - computes a SHA-256 content hash;
    - extracts the optional CIFAKE semantic subclass from the filename.

No dataset-level decisions are made at this stage. Duplicate detection,
cross-split leakage handling, acceptance/rejection decisions, and split
generation are handled by the downstream data integrity stage.

Input:
    data/raw/

Output:
    data/interim/inspected_records.csv

Usage:
    python -m gan_busters.data_pipeline.inspect
"""

from pathlib import Path
import hashlib
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import re
import csv

from loguru import logger
from tqdm import tqdm
import typer

from gan_busters.config import RAW_DATA_DIR, INTERIM_DATA_DIR, CIFAR10_CLASSES

app = typer.Typer()


# extract the split and label info from the directory structure
def parse_image_path(path: Path, input_dir: Path):
    """Extract and validate the split and label from an image path."""
    rel = path.relative_to(input_dir)
    parts = rel.parts # e.g. ("train", "FAKE", "image2.jpg")

    # dataset with no split, e.g. REAL/image.jpg or FAKE/image.jpg
    if len(parts) == 2:
        label = parts[0].upper()

        if label not in {"REAL", "FAKE"}:
            return None

        return None, label

    # dataset with split: train/REAL/image.jpg, test/FAKE/image.jpg, etc.
    if len(parts) == 3:
        split = parts[0].lower()
        label = parts[1].upper()

        if split not in {"train", "test"}:
            return None

        if label not in {"REAL", "FAKE"}:
            return None

        return split, label

    return None


# check if image file can be opened & extract image hash
def inspect_image(path: Path):
    """Validate image readability and compute its SHA-256 content hash."""
    data = path.read_bytes()
    file_hash = hashlib.sha256(data).hexdigest()

    try:
        with Image.open(BytesIO(data)) as img:
            img.load()
        return "valid", file_hash

    except UnidentifiedImageError:
        return "not_image", file_hash # the file is not a valid image format like jpg or png

    except OSError:
        return "corrupted", file_hash

    
# currently the semantic classes are encoded in the filenames in between the ( )
def extract_cifake_subclass(path: Path) -> str | None:
    """Extract the CIFAR-10 semantic class from a CIFAKE filename."""
    match = re.search(r"\((\d+)\)$", path.stem)

    # CIFAKE uses no suffix for the first CIFAR-10 class (airplane)
    class_id = int(match.group(1)) if match else 1

    return CIFAR10_CLASSES.get(class_id)


@app.command()
def main(
    input_dir: Path = RAW_DATA_DIR,
    output_dir: Path = INTERIM_DATA_DIR,
):
    records = []

    for path in tqdm(input_dir.rglob("*"), desc="Inspecting files"):

        # Ignore directories and KaggleHub metadata
        if not path.is_file() or ".complete" in path.parts:
            continue

        rel = path.relative_to(input_dir)

        record = {
            "relative_path": str(rel),
            "split": None,
            "label": None,
            "subclass": None,
            "hash": None,
            "status": None,
        }

        # Inspect directory structure
        parsed = parse_image_path(path, input_dir)

        if parsed is None:
            record["status"] = "invalid_path"
            records.append(record)
            continue

        record["split"], record["label"] = parsed

        # Inspect image contents and compute hash
        image_status, file_hash = inspect_image(path)

        record["hash"] = file_hash
        record["status"] = image_status

        if image_status != "valid":
            records.append(record)
            continue

        # Extract optional semantic class
        record["subclass"] = extract_cifake_subclass(path)

        records.append(record)


    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "inspected_records.csv"

    fields = [
        "relative_path",
        "split",
        "label",
        "subclass",
        "hash",
        "status",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    valid_count = sum(
        record["status"] == "valid"
        for record in records
    )

    logger.success(
        f"Inspection complete: "
        f"{len(records)} files inspected, "
        f"{valid_count} valid files, "
        f"{len(records) - valid_count} files with integrity issues. "
        f"Records saved to {output_path}"
    )


if __name__ == "__main__":
    app()