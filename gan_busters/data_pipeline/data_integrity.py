"""
Apply dataset-level integrity rules to the inspected image records and produce
the canonical dataset definition used by downstream model development.

This stage:
    - separates valid and invalid files based on the inspection results;
    - detects exact duplicate images using content hashes;
    - rejects images with contradictory target labels;
    - prevents cross-split data leakage;
    - removes duplicate observations within the same split;
    - preserves complete predefined train/test splits;
    - generates a reproducible stratified split when no complete split exists.

Input:
    data/interim/inspected_records.csv

Output:
    data/processed/accepted_records.csv
    data/processed/rejected_records.csv

Usage:
    python -m gan_busters.data_pipeline.data_integrity
"""

import csv
from pathlib import Path

from loguru import logger
from sklearn.model_selection import train_test_split
import typer

from gan_busters.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_SEED

app = typer.Typer()


# handle 3 possible cases of duplicates: 
# 1. same content, different labels -> drop ALL copies (contradictory labels)
# 2. same content, same label but different splits -> cross split leakage, keep one copy in the test set only
# 3. same content, same labels, same split -> keep one copy

def resolve_duplicates(
    records: list[dict],
    has_predefined_split: bool,
) -> tuple[list[dict], list[dict]]:
    """Resolve duplicate images based on hash, label, and optional split."""

    # Group all occurrences of the same image
    by_hash = {}

    for record in records:
        by_hash.setdefault(record["hash"], []).append(record)

    accepted = []
    rejected = []

    for duplicates in by_hash.values():

        # No duplicates
        if len(duplicates) == 1:
            accepted.append(duplicates[0])
            continue

        labels = {record["label"] for record in duplicates}

        # Same image assigned different labels -> reject every copy
        if len(labels) > 1:
            for record in duplicates:
                record["reason"] = "conflicting_labels"
                rejected.append(record)
            continue


        if has_predefined_split:
            splits = {record["split"] for record in duplicates}


            # Same image and label across train/test -> keep the test copy
            if "train" in splits and "test" in splits:
                kept = next(
                    record for record in duplicates
                    if record["split"] == "test"
                )

                accepted.append(kept)

                for record in duplicates:
                    if record is not kept:
                        record["reason"] = "cross_split_leakage"
                        rejected.append(record)

                continue


        # Same image and label within the same split,
        # or dataset has no predefined split -> keep one copy
        kept = duplicates[0]
        accepted.append(kept)

        for record in duplicates[1:]:
            record["reason"] = (
                "duplicate_same_split"
                if has_predefined_split
                else "duplicate"
            )
            rejected.append(record)

    return accepted, rejected



def has_complete_split(records: list[dict]) -> bool:
    """Return True if every record has a predefined train/test split."""
    return all(
        record["split"] in {"train", "test"}
        for record in records
    )



def assign_train_test_split(
    records: list[dict],
    test_size: float = 0.2,
) -> None:
    
    """
    Assign a reproducible stratified train/test split in place.

    Stratification always preserves the REAL/FAKE target distribution.
    When semantic subclass information is available for all records,
    stratification additionally preserves the joint label/subclass
    distribution.
    """

    # Use semantic subclass only when available for every record
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

    train_records, test_records = train_test_split(
        records,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )

    for record in train_records:
        record["split"] = "train"

    for record in test_records:
        record["split"] = "test"


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "inspected_records.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
):
    records = []
    rejected = []

    # ---------------------------------------------------------
    # 1. Read inspected records
    # ---------------------------------------------------------
    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for record in reader:

            # Convert empty CSV values back to None
            record = {
                key: value if value != "" else None
                for key, value in record.items()
            }

            record["reason"] = None

            if record["status"] == "valid":
                records.append(record)
            else:
                record["reason"] = record["status"]
                record["status"] = "rejected"
                rejected.append(record)

    logger.info(
        f"Loaded {len(records) + len(rejected)} inspected records: "
        f"{len(records)} valid candidates, "
        f"{len(rejected)} files with integrity issues."
    )

    # ---------------------------------------------------------
    # 2. Check predefined split
    # ---------------------------------------------------------
    complete_split = has_complete_split(records)

    if complete_split:
        logger.info(
            "Complete predefined train/test split detected. "
            "Existing split assignments will be preserved."
        )

    else:
        logger.info(
            "Incomplete or missing predefined split detected. "
            "Existing split assignments will be discarded."
        )

        # If the split is incomplete, discard it entirely before
        # duplicate resolution.
        for record in records:
            record["split"] = None

    # ---------------------------------------------------------
    # 3. Resolve duplicates
    # ---------------------------------------------------------
    accepted, duplicate_rejections = resolve_duplicates(records, has_predefined_split=complete_split)

    rejected.extend(duplicate_rejections)

    logger.info(
        f"Duplicate resolution complete: "
        f"{len(accepted)} images accepted, "
        f"{len(duplicate_rejections)} duplicates/conflicts rejected."
    )

    # ---------------------------------------------------------
    # 4. Generate train/test split when required
    # ---------------------------------------------------------
    if not complete_split:
        assign_train_test_split(
            accepted,
            test_size=0.2,
        )

        logger.info(
            f"New 80/20 stratified train/test split created "
            f"using random seed {RANDOM_SEED}."
        )

    # ---------------------------------------------------------
    # 5. Assign final statuses
    # ---------------------------------------------------------
    for record in accepted:
        record["status"] = "accepted"

    for record in duplicate_rejections:
        record["status"] = "rejected"

    # ---------------------------------------------------------
    # 6. Create output directory
    # ---------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 7. Write accepted records
    # ---------------------------------------------------------
    accepted_path = output_dir / "accepted_records.csv"

    accepted_fields = [
        "relative_path",
        "split",
        "label",
        "subclass",
        "hash",
    ]

    with accepted_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=accepted_fields)
        writer.writeheader()

        for record in accepted:
            writer.writerow({
                field: record[field]
                for field in accepted_fields
            })

    # ---------------------------------------------------------
    # 8. Write rejected records
    # ---------------------------------------------------------
    rejected_path = output_dir / "rejected_records.csv"

    rejected_fields = [
        "relative_path",
        "split",
        "label",
        "subclass",
        "hash",
        "status",
        "reason",
    ]

    with rejected_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rejected_fields)
        writer.writeheader()

        for record in rejected:
            writer.writerow({
                field: record[field]
                for field in rejected_fields
            })

    # ---------------------------------------------------------
    # 9. Final summary
    # ---------------------------------------------------------
    logger.success(
        f"Data integrity checks complete: "
        f"{len(accepted)} records accepted, "
        f"{len(rejected)} records rejected. "
        f"Results saved to {output_dir}"
    )


if __name__ == "__main__":
    app()