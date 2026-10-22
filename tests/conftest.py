import shutil

import pytest
from PIL import Image
import csv


@pytest.fixture
def sample_image(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32)).save(path)
    return path


@pytest.fixture
def sample_non_image(tmp_path):
    path = tmp_path / "invalid.jpg"
    path.write_text("abc", encoding="utf-8")
    return path


@pytest.fixture
def sample_raw_data(tmp_path, sample_image, sample_non_image):
    input_dir = tmp_path / "raw"

    # create dataset directories
    real_dir = input_dir / "train" / "REAL"
    fake_dir = input_dir / "train" / "FAKE"
    val_dir = input_dir / "val" / "REAL"
    invalid_dir = input_dir / "unknown"

    for directory in [real_dir, fake_dir, val_dir, invalid_dir]:
        directory.mkdir(parents=True)

    # Reuse the sample image
    shutil.copy(sample_image, real_dir / "abc123.png")
    shutil.copy(sample_image, fake_dir / "abc456(3).png")
    shutil.copy(sample_image, val_dir / "abc789.png")
    shutil.copy(sample_image, invalid_dir / "abc111.png")

    # Reuse the non-image file
    shutil.copy(sample_non_image, real_dir / "invalid.jpg")

    # kaggleHub metadata, it shouldnt be traced as a file by the recursive funtction
    (input_dir / ".complete").write_text("metadata", encoding="utf-8")

    return input_dir


# helper to generate two dataset fixtures given record(s) value attributes
@pytest.fixture
def sample_inspected_records(tmp_path):
    def create_csv(records):
        path = tmp_path / "inspected_records.csv"

        fields = [
            "relative_path",
            "split",
            "label",
            "subclass",
            "hash",
            "status",
        ]

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

        return path

    return create_csv

