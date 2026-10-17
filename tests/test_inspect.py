from pathlib import Path

import csv
import re
import pytest
from PIL import Image

from gan_busters.data_pipeline import inspect
from gan_busters import config


###### parse_image_path tests ######

@pytest.mark.parametrize(
    "relative_path, expected",
    [
        pytest.param("train/REAL/image.jpg", ("train", "REAL"), id="train-real"),
        pytest.param("test/FAKE/image.jpg", ("test", "FAKE"), id="test-fake"),
        pytest.param("val/REAL/image.png", ("val", "REAL"), id="validation-real"),
        pytest.param("TRAIN/real/image.jpg", ("train", "REAL"), id="case-insensitive"),
        pytest.param("train/REAL/image.png", ("train", "REAL"), id="different-extension"),
        pytest.param("REAL/image.jpg", (None, "REAL"), id="unsplit-real"), #pipeline will assign a split later
        pytest.param("FAKE/image.jpg", (None, "FAKE"), id="unsplit-fake"), #pipeline will assign a split later
        pytest.param("REAL/train/image.jpg", None, id="incorrect-order-real"),
        pytest.param("train/REAL/not_an_image.txt", ("train", "REAL"), id="non-image-extension"), #the pipeline will handle this later
        pytest.param("FAKE/test/image.jpg", None, id="incorrect-order-fake"),
        pytest.param("train/UNKNOWN/image.jpg", None, id="invalid-label"),
        pytest.param("validation/REAL/image.jpg", None, id="invalid-split"),
        pytest.param("UNKNOWN/image.jpg", None, id="invalid-unsplit-label"),
        pytest.param("image.jpg", None, id="missing-directories"),
        pytest.param("train/REAL/subfolder/image.jpg", None, id="extra-directory"),
    ],
)

def test_parse_image_path(relative_path, expected):
    result = inspect.parse_image_path(Path(relative_path))

    assert result == expected



####### inspect_image tests for valid, not image and corrupted images ######
@pytest.mark.parametrize(
    "format, extension, size",
    [
        pytest.param("PNG", ".png", (32, 32), id="valid-png"),
        pytest.param("JPEG", ".jpg", (32, 32), id="valid-jpg"),
        pytest.param("PNG", ".png", (64, 64), id="png-different-size"),
    ],
)
def test_inspect_valid_image(tmp_path, format, extension, size):
    path = tmp_path / f"sample{extension}"
    Image.new("RGB", size).save(path, format=format)

    status, file_hash = inspect.inspect_image(path)

    assert status == "valid"
    assert re.fullmatch(r"[0-9a-f]{64}", file_hash)


def test_inspect_non_image(sample_non_image):
    status, file_hash = inspect.inspect_image(sample_non_image)

    assert status == "not_image"
    assert file_hash == (
        "ba7816bf8f01cfea414140de5dae2223" #known hash values for the "abc" string
        "b00361a396177a9cb410ff61f20015ad"
    )

def test_inspect_corrupted_image(sample_image):
    data = sample_image.read_bytes()

    sample_image.write_bytes(data[:len(data) // 2])

    status, file_hash = inspect.inspect_image(sample_image)

    assert status == "corrupted"
    assert re.fullmatch(r"[0-9a-f]{64}", file_hash)


##### extract_cifake_class tests #####
@pytest.mark.parametrize(
    "path, expected",
    [
        pytest.param("train/REAL/image(3).jpg", config.CIFAR10_CLASSES[3], id="valid-class"),
        pytest.param("train/REAL/image.jpg", config.CIFAR10_CLASSES[1], id="no-suffix-airplane"),
        pytest.param("train/REAL/abc123(3).jpg", config.CIFAR10_CLASSES[3], id="alphanumeric-id"),
        pytest.param("train/REAL/abc_123(3).jpg", None, id="invalid-image-id"),
        pytest.param("train/REAL/image(abc).jpg", None, id="non-numeric-suffix"),
        pytest.param("train/REAL/image(3)extra.jpg", None, id="suffix-not-at-end"),
        pytest.param("train(3)/REAL/image.jpg", config.CIFAR10_CLASSES[1], id="digits-in-directory"),
        pytest.param("train(3)/REAL/image(5).jpg", config.CIFAR10_CLASSES[5], id="directory-ignored"),
        pytest.param("train/REAL/image(99).jpg", None, id="unknown-class-id"),
        pytest.param("train/REAL/image().jpg", None, id="empty-parentheses"),
        pytest.param("train/REAL/image(3)(5).jpg", None, id="multiple-suffixes"),
    ],
)
def test_extract_cifake_subclass(path, expected):
    result = inspect.extract_cifake_subclass(Path(path))

    assert result == expected


### test the entire main functionality with all the functions integrated with a predefined split ### 
def test_inspect_main(sample_raw_data, tmp_path): #uses the sample_raw_data fixture created in conftest
    output_dir = tmp_path / "interim"

    # Run the inspection pipeline
    inspect.main(input_dir=sample_raw_data, output_dir=output_dir)

    # Read the generated CSV
    output_path = output_dir / "inspected_records.csv"
    assert output_path.exists()

    with output_path.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    # Four images + one non-image file (.complete is ignored)
    assert len(records) == 5

    # Index records by relative path for easier assertions
    records_by_path = {
        record["relative_path"]: record
        for record in records
    }

    # KaggleHub metadata should be ignored
    assert ".complete" not in records_by_path

    # Valid REAL image
    real = records_by_path["train/REAL/abc123.png"]
    assert real["status"] == "valid"
    assert real["split"] == "train"
    assert real["label"] == "REAL"
    assert real["subclass"] == config.CIFAR10_CLASSES[1]
    assert re.fullmatch(r"[0-9a-f]{64}", real["hash"])

    # Valid FAKE image
    fake = records_by_path["train/FAKE/abc456(3).png"]
    assert fake["status"] == "valid"
    assert fake["label"] == "FAKE"
    assert fake["subclass"] == config.CIFAR10_CLASSES[3]

    # Valid validation image
    validation = records_by_path["val/REAL/abc789.png"]
    assert validation["status"] == "valid"
    assert validation["split"] == "val"
    assert validation["label"] == "REAL"
    assert validation["subclass"] == config.CIFAR10_CLASSES[1]
    assert re.fullmatch(r"[0-9a-f]{64}", validation["hash"])

    # Non-image file
    non_image = records_by_path["train/REAL/invalid.jpg"]
    assert non_image["status"] == "not_image"
    assert non_image["subclass"] == "" #instead of assert is None, since we are testing the csv output, not the internal dictionary

    # Invalid directory structure
    invalid = records_by_path["unknown/abc111.png"]
    assert invalid["status"] == "invalid_path"
    assert invalid["hash"] == ""


