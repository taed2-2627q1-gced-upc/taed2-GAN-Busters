import csv
from collections import Counter

import pytest

from gan_busters.data_pipeline import split


#### test the first 3 small functions ####
@pytest.mark.parametrize(
    "splits, expected",
    [
        ({"train", "val", "test"}, (True, False, False)),
        ({"train", "test"}, (False, True, False)),
        ({"train", "val"}, (False, False, True)),
        ({None}, (False, False, False)),
        ({"train"}, (False, False, False)),
        ({"train", "test", None}, (False, False, False)),
    ],
)
def test_split_detection(splits, expected):
    result = (
        split.has_complete_split(splits),
        split.has_train_test_split(splits),
        split.has_train_val_split(splits),
    )

    assert result == expected


#### create a small dataset to check how the function assigns the split ####
def make_records(with_subclasses):
    # creating a small dataset
    return [
        {
            "relative_path": f"image_{i}.png",
            "split": None,
            "label": "REAL" if i < 200 else "FAKE",
            "subclass": (
                "cat" if i % 2 == 0 else "dog"
            ) if with_subclasses else None,
        }
        for i in range(400)
    ]

@pytest.mark.parametrize("with_subclasses", [True, False])
def test_assign_split(with_subclasses):

    # creating two equal datasets to see if the random seed will generate the same split
    records1 = make_records(with_subclasses)
    records2 = make_records(with_subclasses)

    split.assign_split(records1, 0.25, "train", "test")
    split.assign_split(records2, 0.25, "train", "test")

    # checking same seed produces identical assignments
    assert [r["split"] for r in records1] == [
        r["split"] for r in records2
    ]

    # checking the distribution is maintained
    assert Counter(r["split"] for r in records1) == {
        "train": 300,
        "test": 100,
    }

    # checking that both classes are represented equally in the test set
    train_records = [r for r in records1 if r["split"] == "train"]
    test_records = [r for r in records1 if r["split"] == "test"]

    train_real_ratio = sum(r["label"] == "REAL" for r in train_records) / len(train_records)
    test_real_ratio = sum(r["label"] == "REAL" for r in test_records) / len(test_records)
    assert abs(train_real_ratio - test_real_ratio) <= 0.05

    train_cat_ratio = sum(r["subclass"] == "cat" for r in train_records) / len(train_records)
    test_cat_ratio = sum(r["subclass"] == "cat" for r in test_records) / len(test_records)
    assert abs(train_cat_ratio - test_cat_ratio) <= 0.05



###### now testing the main function ########
@pytest.mark.parametrize(
    "original_splits",
    [
        {"train", "val", "test"},
        {"train", "test"},
        {"train", "val"},
        {None},
    ],
)
def test_split_main(tmp_path, original_splits):
    input_path = tmp_path / "integrity_records.csv"
    output_path = tmp_path / "accepted_records.csv"

    records = []

    for i in range(400):
        label = "REAL" if i % 2 == 0 else "FAKE"

        if original_splits == {None}:
            assigned = None
        else:
            assigned = sorted(original_splits)[i % len(original_splits)]

        records.append({
            "relative_path": f"image_{i}.png",
            "split": assigned,
            "label": label,
            "subclass": None,
            "hash": f"hash_{i}",
        })

    fields = ["relative_path", "split", "label", "subclass", "hash"]

    with input_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    split.main(input_path=input_path, output_path=output_path)

    assert output_path.exists()

    with output_path.open(newline="", encoding="utf-8") as f:
        result = list(csv.DictReader(f))

    # No records should be lost
    assert len(result) == len(records)

    # All three splits must exist
    assert {r["split"] for r in result} == {"train", "val", "test"}

    # Predefined validation and test assignments must be preserved, not moved 
    for before, after in zip(records, result):
        if before["split"] in {"val", "test"}:
            assert after["split"] == before["split"]