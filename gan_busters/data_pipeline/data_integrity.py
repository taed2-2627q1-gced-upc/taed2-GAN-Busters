"""
Apply dataset-level integrity rules to the inspected image records.

This stage:
    - separates valid and invalid files based on the inspection results;
    - detects exact duplicate images using content hashes;
    - rejects images with contradictory target labels;
    - prevents cross-split data leakage;
    - removes duplicate observations within the same split;
    - preserves complete predefined train/test splits when available.

Input:
    data/interim/inspected_records.csv

Output:
    data/interim/integrity_records.csv
    data/processed/rejected_records.csv

Usage:
    python -m gan_busters.data_pipeline.data_integrity
"""

import csv
from pathlib import Path

from loguru import logger
import typer

from gan_busters import config
from gan_busters.data_pipeline.split import has_complete_split

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
                record["rejection_reason"] = "conflicting_labels"
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
                        record["rejection_reason"] = "cross_split_leakage"
                        rejected.append(record)

                continue


        # Same image and label within the same split,
        # or dataset has no predefined split -> keep one copy
        kept = duplicates[0]
        accepted.append(kept)

        for record in duplicates[1:]:
            record["rejection_reason"] = (
                "duplicate_same_split"
                if has_predefined_split
                else "duplicate"
            )
            rejected.append(record)

    return accepted, rejected


@app.command()
def main(
    input_path: Path = config.INTERIM_DATA_DIR / "inspected_records.csv",
    integrity_path: Path = config.INTERIM_DATA_DIR / "integrity_records.csv",
    rejected_path: Path = config.PROCESSED_DATA_DIR / "rejected_records.csv",
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

            record["rejection_reason"] = None

            if record["status"] == "valid":
                records.append(record)
            else:
                record["rejection_reason"] = record["status"]
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
    integrity_records, duplicate_rejections = resolve_duplicates(records, has_predefined_split=complete_split)

    rejected.extend(duplicate_rejections)

    logger.info(
        f"Duplicate resolution complete: "
        f"{len(integrity_records)} images passed integrity checks, "
        f"{len(duplicate_rejections)} duplicates/conflicts rejected."
    )

    # ---------------------------------------------------------
    # 4. Create output directories
    # ---------------------------------------------------------
    integrity_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 5. Write integrity records
    # ---------------------------------------------------------

    integrity_fields = [
        "relative_path",
        "split",
        "label",
        "subclass",
        "hash",
    ]

    with integrity_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=integrity_fields)
        writer.writeheader()

        for record in integrity_records:
            writer.writerow({
                field: record[field]
                for field in integrity_fields
            })

    # ---------------------------------------------------------
    # 6. Write rejected records
    # ---------------------------------------------------------

    rejected_fields = [
        "relative_path",
        "split",
        "label",
        "subclass",
        "hash",
        "rejection_reason",
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
    # 7. Final summary
    # ---------------------------------------------------------
    logger.success(
        f"Data integrity checks complete: "
        f"{len(integrity_records)} records passed integrity checks and "
        f"were saved to {integrity_path}. "
        f"{len(rejected)} records rejected and "
        f"saved to {rejected_path}."
    )


if __name__ == "__main__":
    app()