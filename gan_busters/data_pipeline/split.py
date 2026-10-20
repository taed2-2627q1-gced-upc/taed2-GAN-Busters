# pylint: disable=duplicate-code
"""
Create reproducible train/validation/test splits from integrity-checked records.

This stage:
    - preserves complete predefined train/validation/test splits;
    - preserves predefined train/test or train/validation splits;
    - generates missing test or validation sets from training records;
    - generates reproducible stratified splits when no usable split exists.

Input:
    data/interim/integrity_records.csv

Output:
    data/processed/accepted_records.csv

Usage:
    python -m gan_busters.data_pipeline.split
"""

import csv
from pathlib import Path

from loguru import logger
from sklearn.model_selection import train_test_split
import typer

from gan_busters import config


app = typer.Typer()


def has_complete_split(splits: set) -> bool:
    """Return True if splits contain train, val, and test."""
    return splits == {"train", "val", "test"}


def has_train_test_split(splits: set) -> bool:
    """Return True if splits contain exactly train and test."""
    return splits == {"train", "test"}


def has_train_val_split(splits: set) -> bool:
    """Return True if splits contain exactly train and val."""
    return splits == {"train", "val"}


def assign_split(
    records: list[dict],
    split_size: float,
    first_split: str,
    second_split: str,
) -> None:
    """Assign a reproducible stratified split in place."""

    has_subclasses = all(
        record["subclass"] is not None
        for record in records
    )

    if has_subclasses:
        stratify = [
            f"{record['label']}::{record['subclass']}"
            for record in records
        ]
    else:
        stratify = [
            record["label"]
            for record in records
        ]

    first_records, second_records = train_test_split(
        records,
        test_size=split_size,
        random_state=config.RANDOM_SEED,
        stratify=stratify,
    )

    for record in first_records:
        record["split"] = first_split

    for record in second_records:
        record["split"] = second_split


@app.command()
def main(
    input_path: Path = config.INTERIM_DATA_DIR / "integrity_records.csv",
    output_path: Path = config.ACCEPTED_RECORDS_PATH,
):
    """Create reproducible train/validation/test splits."""
    records = []

    # ---------------------------------------------------------
    # 1. Read integrity records
    # ---------------------------------------------------------
    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for record in reader:
            record = {
                key: value if value != "" else None
                for key, value in record.items()
            }
            records.append(record)

    logger.info(
        f"Loaded {len(records)} integrity-checked records."
    )

    # ---------------------------------------------------------
    # 2. Preserve or generate train/val/test splits
    # ---------------------------------------------------------
    splits = {record["split"] for record in records}

    if has_complete_split(splits):
        logger.info(
            "Complete predefined train/val/test split detected. "
            "All assignments preserved."
        )

    elif has_train_test_split(splits):
        # Preserve test, generate validation from train
        train_records = [
            record for record in records
            if record["split"] == "train"
        ]

        assign_split(
            train_records,
            split_size=config.DEFAULT_VALIDATION_SIZE,
            first_split="train",
            second_split="val",
        )

        logger.info(
            "Predefined train/test split preserved. "
            "Validation generated from training records."
        )

    elif has_train_val_split(splits):
        # Preserve validation, generate test from train
        train_records = [
            record for record in records
            if record["split"] == "train"
        ]

        assign_split(
            train_records,
            split_size=config.DEFAULT_TEST_SIZE,
            first_split="train",
            second_split="test",
        )

        logger.info(
            "Predefined train/val split preserved. "
            "Test generated from training records."
        )

    else:
        # No usable predefined split: generate all three
        for record in records:
            record["split"] = None

        assign_split(
            records,
            split_size=config.DEFAULT_TEST_SIZE,
            first_split="train",
            second_split="test",
        )

        train_records = [
            record for record in records
            if record["split"] == "train"
        ]

        assign_split(
            train_records,
            split_size=config.DEFAULT_VALIDATION_SIZE,
            first_split="train",
            second_split="val",
        )

        logger.info(
            "No usable predefined split. "
            "New stratified train/val/test splits generated."
        )

    # ---------------------------------------------------------
    # 3. Write canonical dataset records
    # ---------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "relative_path",
        "split",
        "label",
        "subclass",
        "hash",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for record in records:
            writer.writerow({
                field: record[field]
                for field in fields
            })

    # ---------------------------------------------------------
    # 4. Final summary
    # ---------------------------------------------------------
    train_count = sum(
        record["split"] == "train"
        for record in records
    )
    val_count = sum(
        record["split"] == "val"
        for record in records
    )
    test_count = sum(
        record["split"] == "test"
        for record in records
    )

    logger.success(
        f"Dataset split complete: "
        f"{train_count} train, "
        f"{val_count} validation, "
        f"{test_count} test records. "
        f"Saved to {output_path}."
    )


if __name__ == "__main__":
    app()
