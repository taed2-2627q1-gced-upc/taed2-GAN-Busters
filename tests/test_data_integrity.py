import pytest
import csv
from collections import Counter
from gan_busters.data_pipeline import data_integrity


def make_record(path, split, label, hash_value, status="valid"):
    return {
        "relative_path": path,
        "split": split,
        "label": label,
        "subclass": "automobile" if status == "valid" else None, 
        "hash": hash_value,
        "status": status,
    }


#### test resolve_duplicates when the split column is complete ####
@pytest.mark.parametrize(
    "records, expected_accepted_paths, expected_rejected",
    [
        pytest.param(
            [
                make_record("train/REAL/a.png", "train", "REAL", "hash_A"),
            ],
            ["train/REAL/a.png"],
            {},
            id="unique-image",
        ),
        pytest.param(
            [
                make_record("train/REAL/a.png", "train", "REAL", "hash_A"),
                make_record("train/REAL/b.png", "train", "REAL", "hash_A"),
            ],
            ["train/REAL/a.png"],
            {"duplicate_same_split": 1},
            id="same-split-duplicates",
        ),
        pytest.param(
            [
                make_record("train/REAL/a.png", "train", "REAL", "hash_A"),
                make_record("test/REAL/b.png", "test", "REAL", "hash_A"),
            ],
            ["test/REAL/b.png"],
            {"cross_split_leakage": 1},
            id="cross-split-leakage",
        ),
        pytest.param(
            [
                make_record("train/REAL/a.png", "train", "REAL", "hash_A"),
                make_record("train/FAKE/b.png", "train", "FAKE", "hash_A"),
            ],
            [],
            {"conflicting_labels": 2},
            id="conflicting-labels-same-split",
        ),
        pytest.param(
            [
                make_record("train/REAL/a.png", "train", "REAL", "hash_A"),
                make_record("test/FAKE/b.png", "test", "FAKE", "hash_A"),
            ],
            [],
            {"conflicting_labels": 2},
            id="conflicting-labels-cross-split",
        ),
        pytest.param(
            [
                make_record("train/REAL/a.png", "train", "REAL", "hash_A"),
                make_record("val/REAL/b.png", "val", "REAL", "hash_A"),
            ],
            ["val/REAL/b.png"],
            {"cross_split_leakage": 1},
            id="train-val-leakage",
        ),
        pytest.param(
            [
                make_record("val/REAL/a.png", "val", "REAL", "hash_A"),
                make_record("test/REAL/b.png", "test", "REAL", "hash_A"),
            ],
            ["test/REAL/b.png"],
            {"cross_split_leakage": 1},
            id="val-test-leakage",
        ),
        pytest.param(
            [
                make_record("train/REAL/a.png", "train", "REAL", "hash_A"),
                make_record("val/REAL/b.png", "val", "REAL", "hash_A"),
                make_record("test/REAL/c.png", "test", "REAL", "hash_A"),
            ],
            ["test/REAL/c.png"],
            {"cross_split_leakage": 2},
            id="train-val-test-leakage",
        ),
        pytest.param(
            [
                make_record("train/REAL/a.png", "train", "REAL", "hash_A"),
                make_record("val/REAL/b.png", "val", "REAL", "hash_A"),
                make_record("test/REAL/c.png", "test", "REAL", "hash_A"),
                make_record("test/REAL/d.png", "test", "REAL", "hash_A"),
            ],
            ["test/REAL/c.png"],
            {
                "cross_split_leakage": 2,
                "duplicate_same_split": 1,
            },
            id="cross-split-and-same-split-duplicates",
        ),
        pytest.param(
            [
                make_record("val/REAL/a.png", "val", "REAL", "hash_A"),
                make_record("val/REAL/b.png", "val", "REAL", "hash_A"),
            ],
            ["val/REAL/a.png"],
            {"duplicate_same_split": 1},
            id="same-val-split-duplicates",
        ),
    ],
)
def test_resolve_duplicates_with_split(
    records, expected_accepted_paths, expected_rejected
):
    accepted, rejected = data_integrity.resolve_duplicates(
        records, has_predefined_split=True
    )

    accepted_paths = [record["relative_path"] for record in accepted]
    assert accepted_paths == expected_accepted_paths

    reasons = Counter(
        record["rejection_reason"] for record in rejected
    )
    assert reasons == expected_rejected


### test resolve duplicates when the split is not provided ####
@pytest.mark.parametrize(
    "records, expected_accepted_paths, expected_rejected",
    [
        pytest.param(
            [
                make_record("REAL/a.png", None, "REAL", "hash_A"),
                make_record("FAKE/b.png", None, "FAKE", "hash_B"),
            ],
            ["REAL/a.png", "FAKE/b.png"],
            {},
            id="different-images-different-labels",
        ),
        pytest.param(
            [
                make_record("REAL/a.png", None, "REAL", "hash_A"),
                make_record("REAL/b.png", None, "REAL", "hash_A"),
            ],
            ["REAL/a.png"],
            {"duplicate": 1},
            id="same-image-same-label",
        ),
        pytest.param(
            [
                make_record("REAL/a.png", None, "REAL", "hash_A"),
                make_record("REAL/b.png", None, "REAL", "hash_A"),
                make_record("REAL/c.png", None, "REAL", "hash_A"),
            ],
            ["REAL/a.png"],
            {"duplicate": 2},
            id="multiple-duplicates",
        ),
        pytest.param(
            [
                make_record("REAL/a.png", None, "REAL", "hash_A"),
                make_record("FAKE/b.png", None, "FAKE", "hash_A"),
            ],
            [],
            {"conflicting_labels": 2},
            id="same-image-different-labels",
        ),
    ],
)
def test_resolve_duplicates_without_split(
    records, expected_accepted_paths, expected_rejected
):
    accepted, rejected = data_integrity.resolve_duplicates(
        records, has_predefined_split=False
    )

    accepted_paths = [record["relative_path"] for record in accepted]
    assert accepted_paths == expected_accepted_paths

    reasons = Counter(
        record["rejection_reason"] for record in rejected
    )
    assert reasons == expected_rejected


#### now lets check that the main() function works on a csv with complete split information ####
def test_main_with_complete_split(sample_inspected_records, tmp_path):
    records = [
        # Unique images
        make_record("train/REAL/a.png", "train", "REAL", "hash_A"),
        make_record("test/FAKE/b.png", "test", "FAKE", "hash_B"),

        # Same-split duplicates
        make_record("train/REAL/c.png", "train", "REAL", "hash_C"),
        make_record("train/REAL/d.png", "train", "REAL", "hash_C"),

        # Cross-split leakage
        make_record("train/FAKE/e.png", "train", "FAKE", "hash_D"),
        make_record("test/FAKE/f.png", "test", "FAKE", "hash_D"),

        # Conflicting labels
        make_record("train/REAL/g.png", "train", "REAL", "hash_E"),
        make_record("test/FAKE/h.png", "test", "FAKE", "hash_E"),

        # Invalid inspection statuses
        make_record("train/REAL/invalid.jpg", "train", "REAL", "hash_F", "corrupted"),
        make_record("unknown/file.png", None, None, None, "invalid_path"),

        # Validation records
        make_record("val/REAL/i.png", "val", "REAL", "hash_I"),  # Unique validation image

        # Train/validation leakage
        make_record("train/REAL/j.png", "train", "REAL", "hash_J"),
        make_record("val/REAL/k.png", "val", "REAL", "hash_J"),

        # Validation/test leakage
        make_record("val/FAKE/l.png", "val", "FAKE", "hash_L"),
        make_record("test/FAKE/m.png", "test", "FAKE", "hash_L"),

        # Same-split duplicate alongside cross-split leakage
        make_record("test/REAL/n.png", "test", "REAL", "hash_N"),
        make_record("test/REAL/o.png", "test", "REAL", "hash_N"),
        make_record("val/REAL/p.png", "val", "REAL", "hash_N"),
    ]

    input_path = sample_inspected_records(records)
    integrity_path = tmp_path / "interim" / "integrity_records.csv"
    rejected_path = tmp_path / "processed" / "rejected_records.csv"

    data_integrity.main(
        input_path=input_path,
        integrity_path=integrity_path,
        rejected_path=rejected_path,
    )

    assert input_path.exists()
    assert integrity_path.exists()
    assert rejected_path.exists()

    # inspect the results in integrity path and rejected path after the records go through main
    with integrity_path.open(newline="", encoding="utf-8") as f:
        accepted = list(csv.DictReader(f))

    with rejected_path.open(newline="", encoding="utf-8") as f:
        rejected = list(csv.DictReader(f))

    accepted_by_path = {
        record["relative_path"]: record for record in accepted
    }

    assert set(accepted_by_path) == {
        "train/REAL/a.png",
        "test/FAKE/b.png",
        "train/REAL/c.png",
        "test/FAKE/f.png",
        "val/REAL/i.png",
        "val/REAL/k.png",
        "test/FAKE/m.png",
        "test/REAL/n.png",
    }

    # Original split assignments are preserved
    assert accepted_by_path["train/REAL/a.png"]["split"] == "train"
    assert accepted_by_path["val/REAL/k.png"]["split"] == "val"
    assert accepted_by_path["test/FAKE/f.png"]["split"] == "test"

    rejected_by_path = {
        record["relative_path"]: record["rejection_reason"]
        for record in rejected
    }

    assert rejected_by_path == {
        "train/REAL/d.png": "duplicate_same_split",
        "train/FAKE/e.png": "cross_split_leakage",
        "train/REAL/g.png": "conflicting_labels",
        "test/FAKE/h.png": "conflicting_labels",
        "train/REAL/invalid.jpg": "corrupted",
        "unknown/file.png": "invalid_path",
        "train/REAL/j.png": "cross_split_leakage",
        "val/FAKE/l.png": "cross_split_leakage",
        "test/REAL/o.png": "duplicate_same_split",
        "val/REAL/p.png": "cross_split_leakage",
    }



#### same as above but with an incomplete split, the function will handle it as if there was no split provided ####
def test_main_with_incomplete_split(sample_inspected_records, tmp_path):
    records = [
        # Unique images
        make_record("train/REAL/a.png", "train", "REAL", "hash_A"),
        make_record("FAKE/b.png", None, "FAKE", "hash_B"), # this will make has_complete_split fail

        # Same-split duplicates
        make_record("train/REAL/c.png", "train", "REAL", "hash_C"),
        make_record("train/REAL/d.png", "train", "REAL", "hash_C"),

        # Cross-split duplicates (should become ordinary duplicates)
        make_record("train/FAKE/e.png", "train", "FAKE", "hash_D"),
        make_record("test/FAKE/f.png", "test", "FAKE", "hash_D"),

        # Conflicting labels within the same split
        make_record("train/REAL/g.png", "train", "REAL", "hash_E"),
        make_record("train/FAKE/h.png", "train", "FAKE", "hash_E"),

        # Conflicting labels across splits
        make_record("train/REAL/i.png", "train", "REAL", "hash_F"),
        make_record("test/FAKE/j.png", "test", "FAKE", "hash_F"),

        # Invalid images
        make_record("train/REAL/invalid.jpg", "train", "REAL", "hash_G", "corrupted"),
        make_record("unknown/file.png", None, None, None, "invalid_path"),

        # Train/validation duplicates
        make_record("train/REAL/k.png", "train", "REAL", "hash_H"),
        make_record("val/REAL/l.png", "val", "REAL", "hash_H"),

        # Validation/test duplicates
        make_record("val/FAKE/m.png", "val", "FAKE", "hash_I"),
        make_record("test/FAKE/n.png", "test", "FAKE", "hash_I"),

        # Unique validation image
        make_record("val/REAL/o.png", "val", "REAL", "hash_J"),
    ]

    input_path = sample_inspected_records(records)
    integrity_path = tmp_path / "interim" / "integrity_records.csv"
    rejected_path = tmp_path / "processed" / "rejected_records.csv"

    data_integrity.main(
        input_path=input_path,
        integrity_path=integrity_path,
        rejected_path=rejected_path,
    )

    assert input_path.exists()
    assert integrity_path.exists()
    assert rejected_path.exists()

    with integrity_path.open(newline="", encoding="utf-8") as f:
        accepted = list(csv.DictReader(f))

    with rejected_path.open(newline="", encoding="utf-8") as f:
        rejected = list(csv.DictReader(f))

    # Verify accepted records
    accepted_by_path = {
        record["relative_path"]: record for record in accepted
    }

    assert set(accepted_by_path) == {
        "train/REAL/a.png",
        "FAKE/b.png",
        "train/REAL/c.png",
        "train/FAKE/e.png",
        "train/REAL/k.png",
        "val/FAKE/m.png",
        "val/REAL/o.png",
    }

    # All accepted records should have their split assignments removed
    assert all(record["split"] == "" for record in accepted)

    # Verify rejected records and their reasons
    rejected_by_path = {
        record["relative_path"]: record["rejection_reason"]
        for record in rejected
    }

    assert rejected_by_path == {
        "train/REAL/d.png": "duplicate",
        "test/FAKE/f.png": "duplicate",
        "train/REAL/g.png": "conflicting_labels",
        "train/FAKE/h.png": "conflicting_labels",
        "train/REAL/i.png": "conflicting_labels",
        "test/FAKE/j.png": "conflicting_labels",
        "train/REAL/invalid.jpg": "corrupted",
        "unknown/file.png": "invalid_path",
        "val/REAL/l.png": "duplicate",
        "test/FAKE/n.png": "duplicate",
    }

    # The rejected images should have their split cleared if they reached duplicate checks
    assert all(record["split"] == "" for record in rejected if record["rejection_reason"] in {"duplicate", "conflicting_labels"})