"""
Image data testing and distribution verification with Deepchecks Vision.

Uses accepted_records.csv as the canonical dataset index. Images are read
from data/raw/ using the relative paths stored in the CSV, so both
`dvc pull` (for the CSV) and `python -m gan_busters.data_pipeline.download`
(for the images) must have been run first.

Two suites are executed on stratified samples (label x subclass):
    1. Data integrity on the train split (image property outliers and
       property/label correlation, i.e. "does a simple image statistic leak
       the label?").
    2. Train vs test validation (label drift, image property drift, image
       dataset drift, new labels, ...). Disable with --no-drift.

Outputs:
    reports/deepchecks_data_integrity_report.html
    reports/deepchecks_train_test_report.html

Usage:
    python -m gan_busters.data_testing.deepchecks_validation
    python -m gan_busters.data_testing.deepchecks_validation -s 0 --fail-on-issues
"""

from collections import Counter
from pathlib import Path
from typing import Annotated

from deepchecks.vision import BatchOutputFormat, Suite, VisionData
from deepchecks.vision.checks import ImagePropertyOutliers, PropertyLabelCorrelation
from deepchecks.vision.suites import train_test_validation
from loguru import logger
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import typer

from gan_busters import config

app = typer.Typer(help="Execute statistical data testing using Deepchecks Vision.")

REQUIRED_COLUMNS = {"relative_path", "label", "split"}
DEFAULT_INTEGRITY_REPORT = config.REPORTS_DIR / "deepchecks_data_integrity_report.html"
DEFAULT_DRIFT_REPORT = config.REPORTS_DIR / "deepchecks_train_test_report.html"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
def resolve_image_path(relative_path: str) -> Path | None:
    """Resolve a CSV image path against the raw data and project roots."""
    path = Path(relative_path)
    candidates = (
        [path]
        if path.is_absolute()
        else [config.RAW_DATA_DIR / path, config.PROJ_ROOT / path]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def normalize_label(value) -> int:
    """Convert a CSV label (REAL/FAKE, or 0/1) to an integer class id."""
    if pd.isna(value):
        raise ValueError("Dataset contains a missing label.")

    if isinstance(value, str):
        text = value.strip().upper()
        if text in config.LABEL_MAPPING:
            return config.LABEL_MAPPING[text]
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"Unknown dataset label: {value!r}") from exc

    label = int(value)
    if label not in set(config.LABEL_MAPPING.values()):
        raise ValueError(f"Unsupported numeric label: {label}")
    return label


def load_split(records: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return the records of one split with integer labels."""
    subset = records[records["split"].astype(str).str.lower() == split].copy()
    if subset.empty:
        raise ValueError(f"No records with split='{split}' in accepted_records.csv.")
    subset["label"] = subset["label"].map(normalize_label)
    return subset


def stratified_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """
    Pick at most `sample_size` rows preserving the label x subclass proportions.

    sample_size == 0 (or >= len(df)) keeps every row. Rows are returned shuffled.
    """
    if sample_size < 0:
        raise ValueError("--sample-size must be zero or positive.")

    if sample_size == 0 or sample_size >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if "subclass" in df.columns:
        strata = df["label"].astype(str) + "::" + df["subclass"].fillna("unknown").astype(str)
    else:
        strata = df["label"].astype(str)

    counts = strata.value_counts()
    exact = counts * sample_size / len(df)
    allocation = np.floor(exact).astype(int)

    # Largest-remainder rounding so the allocations add up to sample_size.
    remaining = sample_size - int(allocation.sum())
    if remaining > 0:
        fractional = (exact - allocation).sort_values(ascending=False)
        allocation[fractional.index[:remaining]] += 1
    allocation = np.minimum(allocation, counts)

    parts = [
        group.sample(n=int(allocation[key]), random_state=seed)
        for key, group in df.groupby(strata)
        if allocation[key] > 0
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def attach_image_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve every image up front so a missing download fails fast and clearly."""
    resolved = df["relative_path"].astype(str).map(resolve_image_path)
    missing = df.loc[resolved.isna(), "relative_path"]
    if len(missing):
        raise FileNotFoundError(
            f"{len(missing)} of {len(df)} images listed in accepted_records.csv "
            f"were not found under {config.RAW_DATA_DIR} (first: {missing.iloc[0]}).\n"
            "Run `python -m gan_busters.data_pipeline.download` to fetch data/raw/."
        )
    return df.assign(image_path=resolved.values)


# ---------------------------------------------------------------------------
# Deepchecks data
# ---------------------------------------------------------------------------
class BatchLoader:
    """Iterate over records yielding Deepchecks-compatible classification batches."""

    def __init__(self, records: pd.DataFrame, batch_size: int = 32):
        self.records = records.reset_index(drop=True)
        self.batch_size = batch_size

    def __len__(self) -> int:
        # Lets Deepchecks know the number of batches (correct progress bars).
        return -(-len(self.records) // self.batch_size)

    def __iter__(self):
        for start in range(0, len(self.records), self.batch_size):
            batch = self.records.iloc[start : start + self.batch_size]
            images = []
            for image_path in batch["image_path"]:
                try:
                    with Image.open(image_path) as image:
                        images.append(np.asarray(image.convert("RGB")))
                except (OSError, SyntaxError, UnidentifiedImageError) as exc:
                    raise ValueError(f"Cannot read image {image_path}: {exc}") from exc

            yield BatchOutputFormat(
                images=np.asarray(images, dtype=np.uint8),
                labels=batch["label"].to_numpy(dtype=np.int64),
            )


def build_vision_data(df: pd.DataFrame, name: str, batch_size: int) -> VisionData:
    vision_data = VisionData(
        batch_loader=BatchLoader(df, batch_size=batch_size),
        task_type="classification",
        label_map={v: k for k, v in config.LABEL_MAPPING.items()},
        dataset_name=name,
        reshuffle_data=False,
    )
    logger.info("Validating the Deepchecks loader for '{}'...", name)
    vision_data.validate()
    return vision_data


def prepare_dataset(
    records: pd.DataFrame, split: str, sample_size: int, batch_size: int
) -> VisionData:
    subset = load_split(records, split)
    logger.info(
        "[{}] available: {} records. Class counts: {}",
        split,
        len(subset),
        dict(Counter(subset["label"])),
    )
    selected = attach_image_paths(stratified_sample(subset, sample_size, config.RANDOM_SEED))
    logger.info(
        "[{}] selected: {} records. Class counts: {}",
        split,
        len(selected),
        dict(Counter(selected["label"])),
    )
    return build_vision_data(selected, f"CIFAKE {split} sample", batch_size)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def save_report(result, path: Path) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving HTML report to {}", path)
    result.save_as_html(str(path))

    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Deepchecks did not create a non-empty report: {path}")

    logger.success("Report generated: {} ({} bytes)", path, path.stat().st_size)
    return path


def summarize(result, title: str) -> int:
    """Log condition results; return how many checks did not pass."""
    passed = result.get_passed_checks()
    failed = result.get_not_passed_checks()
    not_run = result.get_not_ran_checks()

    logger.info(
        "{}: {} passed, {} not passed, {} not run.",
        title,
        len(passed),
        len(failed),
        len(not_run),
    )
    for check in failed:
        for condition in check.conditions_results:
            if not condition.is_pass():
                logger.warning(
                    "[{}] {} -> {} ({})",
                    title,
                    check.get_header(),
                    condition.name,
                    condition.details,
                )
    for check in not_run:
        logger.info("[{}] not applicable/not run: {}", title, check.get_header())
    return len(failed)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def run_data_testing(
    sample_size: int = 1000,
    output_report_path: Path = DEFAULT_INTEGRITY_REPORT,
    drift_report_path: Path = DEFAULT_DRIFT_REPORT,
    run_drift: bool = True,
    pps_threshold: float = 0.3,
    batch_size: int = 32,
    fail_on_issues: bool = False,
) -> bool:
    """Run the Deepchecks suites. Returns True when no condition failed."""
    accepted_csv = config.ACCEPTED_RECORDS_PATH
    if not accepted_csv.is_file():
        raise FileNotFoundError(
            f"Dataset index not found: {accepted_csv}\n"
            "Run `python -m dvc pull` (or `python -m dvc repro`) first."
        )

    logger.info("Reading dataset index from {}", accepted_csv)
    records = pd.read_csv(accepted_csv)
    missing_columns = REQUIRED_COLUMNS - set(records.columns)
    if missing_columns:
        raise ValueError(f"{accepted_csv} is missing columns: {sorted(missing_columns)}")

    issues = 0

    # 1. Data integrity on the training sample.
    train_data = prepare_dataset(records, "train", sample_size, batch_size)
    integrity_suite = Suite(
        "Data integrity",
        ImagePropertyOutliers(),
        PropertyLabelCorrelation().add_condition_property_pps_less_than(pps_threshold),
    )
    logger.info("Running the Deepchecks data integrity suite...")
    result = integrity_suite.run(train_dataset=train_data, run_single_dataset="Train")
    save_report(result, output_report_path)
    issues += summarize(result, "integrity")

    # 2. Train vs test distribution validation.
    if run_drift:
        test_data = prepare_dataset(records, "test", sample_size, batch_size)
        logger.info("Running the Deepchecks train/test validation suite...")
        drift_result = train_test_validation().run(
            train_dataset=train_data, test_dataset=test_data
        )
        save_report(drift_result, drift_report_path)
        issues += summarize(drift_result, "train_vs_test")

    if issues:
        logger.warning("{} check(s) did not pass. Review the HTML reports.", issues)
    else:
        logger.success("All Deepchecks conditions passed.")

    return issues == 0 or not fail_on_issues


@app.command()
def main(
    sample_size: Annotated[
        int,
        typer.Option(
            "--sample-size",
            "-s",
            help="Records sampled per split (stratified by label and subclass); 0 means all.",
        ),
    ] = 1000,
    output_path: Annotated[
        Path,
        typer.Option("--output-path", "-o", help="HTML report of the integrity suite."),
    ] = DEFAULT_INTEGRITY_REPORT,
    drift_output_path: Annotated[
        Path,
        typer.Option("--drift-output-path", help="HTML report of the train/test suite."),
    ] = DEFAULT_DRIFT_REPORT,
    drift: Annotated[
        bool,
        typer.Option("--drift/--no-drift", help="Also compare train against test."),
    ] = True,
    pps_threshold: Annotated[
        float,
        typer.Option(
            "--pps-threshold",
            help="Max Predictive Power Score of an image property for the label.",
        ),
    ] = 0.3,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Images per batch.")] = 32,
    fail_on_issues: Annotated[
        bool,
        typer.Option(
            "--fail-on-issues",
            help="Exit with code 1 if any Deepchecks condition does not pass (for CI).",
        ),
    ] = False,
):
    """Run Deepchecks data validation."""
    ok = run_data_testing(
        sample_size=sample_size,
        output_report_path=output_path,
        drift_report_path=drift_output_path,
        run_drift=drift,
        pps_threshold=pps_threshold,
        batch_size=batch_size,
        fail_on_issues=fail_on_issues,
    )
    if not ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()