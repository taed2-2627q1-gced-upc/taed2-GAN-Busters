"""
Great Expectations validation of the dataset class proportions.

Run this stage right after `dvc pull` / `dvc repro`. It only needs
data/processed/accepted_records.csv (the images in data/raw/ are NOT read).

What is verified
----------------
1. Schema and values of accepted_records.csv (columns, no nulls, valid
   label/split/subclass values, exactly the 10 CIFAR-10 subclasses, unique
   hashes and paths).
2. The proportions are the same in train / val / test (and therefore in the
   whole dataset):
     - label:            REAL vs FAKE
     - subclass:         the 10 internal classes (airplane ... truck)
     - label x subclass: the 20 joint cells (e.g. FAKE/cat)
3. REAL and FAKE have the same internal 10-class distribution.

"Same proportion" is checked with an absolute tolerance (in proportion
points, 0.01 = 1 percentage point). It cannot be exactly 0: the integrity
stage removes duplicates (mostly FAKE) and keeps cross-split duplicates only
in the test split, so tiny differences are expected and legitimate.

Outputs
-------
    reports/tables/class_proportions.csv
    reports/great_expectations_class_proportions.json

Usage:
    python -m gan_busters.data_testing.ge_validation
    python -m gan_busters.data_testing.ge_validation --tolerance 0.01
"""

import json
from pathlib import Path
from typing import Annotated

import great_expectations as gx
from great_expectations.data_context.types.base import ProgressBarsConfig
import great_expectations.expectations as gxe
from loguru import logger
import pandas as pd
import typer

from gan_busters import config

app = typer.Typer(help="Validate class proportions with Great Expectations.")

EXPECTED_COLUMNS = ["relative_path", "split", "label", "subclass", "hash"]
LABELS = sorted(config.LABEL_MAPPING)  # ["FAKE", "REAL"]
SPLITS = ["train", "val", "test"]
SUBCLASSES = list(config.CIFAR10_CLASSES.values())

DEFAULT_TOLERANCE = 0.02
DEFAULT_REPORT_PATH = config.REPORTS_DIR / "great_expectations_class_proportions.json"
DEFAULT_TABLE_PATH = config.TABLES_DIR / "class_proportions.csv"


# ---------------------------------------------------------------------------
# Proportion tables
# ---------------------------------------------------------------------------
def _split_vs_global(df: pd.DataFrame, group: pd.Series, categories: list[str]) -> pd.DataFrame:
    """Share of each category inside every split, compared with the global share."""
    counts = pd.crosstab(df["split"], group).reindex(
        index=SPLITS, columns=categories, fill_value=0
    )
    global_counts = counts.sum(axis=0)
    global_share = global_counts / global_counts.sum()

    rows = []
    for split in SPLITS:
        split_total = counts.loc[split].sum()
        for category in categories:
            share = counts.loc[split, category] / split_total if split_total else float("nan")
            rows.append(
                {
                    "split": split,
                    "group": category,
                    "count": int(counts.loc[split, category]),
                    "proportion": share,
                    "reference": global_share[category],
                    "abs_deviation": abs(share - global_share[category]),
                }
            )
    return pd.DataFrame(rows)


def _real_vs_fake(df: pd.DataFrame) -> pd.DataFrame:
    """P(subclass | REAL) compared with P(subclass | FAKE), per split and overall."""
    rows = []
    for scope in [*SPLITS, "all"]:
        sub = df if scope == "all" else df[df["split"] == scope]
        counts = pd.crosstab(sub["subclass"], sub["label"]).reindex(
            index=SUBCLASSES, columns=LABELS, fill_value=0
        )
        totals = counts.sum(axis=0)
        for subclass in SUBCLASSES:
            real = (
                counts.loc[subclass, "REAL"] / totals["REAL"] if totals["REAL"] else float("nan")
            )
            fake = (
                counts.loc[subclass, "FAKE"] / totals["FAKE"] if totals["FAKE"] else float("nan")
            )
            rows.append(
                {
                    "split": scope,
                    "group": subclass,
                    "count": int(counts.loc[subclass].sum()),
                    "proportion": real,  # P(subclass | REAL)
                    "reference": fake,  # P(subclass | FAKE)
                    "abs_deviation": abs(real - fake),
                }
            )
    return pd.DataFrame(rows)


def build_proportion_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One long-format table per proportion check."""
    joint = df["label"] + "/" + df["subclass"]
    joint_categories = [f"{label}/{sub}" for label in LABELS for sub in SUBCLASSES]

    return {
        "label_by_split": _split_vs_global(df, df["label"], LABELS),
        "subclass_by_split": _split_vs_global(df, df["subclass"], SUBCLASSES),
        "label_subclass_by_split": _split_vs_global(df, joint, joint_categories),
        "subclass_real_vs_fake": _real_vs_fake(df),
    }


# ---------------------------------------------------------------------------
# Great Expectations
# ---------------------------------------------------------------------------
class Validator:
    """Small wrapper that runs expectations against pandas DataFrames."""

    def __init__(self) -> None:
        self.context = gx.get_context(mode="ephemeral")
        # Silence the "Calculating Metrics" progress bars (one per expectation).
        self.context.variables.progress_bars = ProgressBarsConfig(
            globally=False, metric_calculations=False
        )
        self.source = self.context.data_sources.add_pandas("class_proportions")
        self.results: list[dict] = []

    def run(self, check: str, frame: pd.DataFrame, expectations: list) -> None:
        """Run expectations against a DataFrame and store their validation results."""
        asset = self.source.add_dataframe_asset(name=check)
        batch_definition = asset.add_batch_definition_whole_dataframe(f"{check}_batch")
        batch = batch_definition.get_batch(batch_parameters={"dataframe": frame})

        for expectation in expectations:
            outcome = batch.validate(expectation).to_json_dict()
            self.results.append(
                {
                    "check": check,
                    "expectation": outcome["expectation_config"]["type"],
                    "kwargs": outcome["expectation_config"]["kwargs"],
                    "success": bool(outcome["success"]),
                    "result": outcome.get("result", {}),
                }
            )

    @property
    def failures(self) -> list[dict]:
        """Return validation results for expectations that did not pass."""        
        return [r for r in self.results if not r["success"]]


def records_expectations() -> list:
    """Define expectations for dataset schema, values, and record uniqueness."""
    expectations = [
        gxe.ExpectTableColumnsToMatchSet(column_set=EXPECTED_COLUMNS, exact_match=True),
        gxe.ExpectTableRowCountToBeBetween(min_value=1),
    ]
    expectations += [gxe.ExpectColumnValuesToNotBeNull(column=c) for c in EXPECTED_COLUMNS]
    expectations += [
        gxe.ExpectColumnValuesToBeInSet(column="label", value_set=LABELS),
        gxe.ExpectColumnValuesToBeInSet(column="split", value_set=SPLITS),
        gxe.ExpectColumnValuesToBeInSet(column="subclass", value_set=SUBCLASSES),
        gxe.ExpectColumnDistinctValuesToEqualSet(column="subclass", value_set=SUBCLASSES),
        gxe.ExpectColumnDistinctValuesToEqualSet(column="split", value_set=SPLITS),
        gxe.ExpectColumnDistinctValuesToEqualSet(column="label", value_set=LABELS),
        gxe.ExpectColumnValuesToBeUnique(column="hash"),
        gxe.ExpectColumnValuesToBeUnique(column="relative_path"),
    ]
    return expectations


def proportion_expectations(tolerance: float) -> list:
    """Define expectations for valid counts and proportion deviations."""
    return [
        gxe.ExpectColumnValuesToNotBeNull(column="abs_deviation"),
        gxe.ExpectColumnValuesToBeBetween(
            column="abs_deviation", min_value=0.0, max_value=tolerance
        ),
        gxe.ExpectColumnValuesToBeBetween(column="count", min_value=1),
    ]


def run_validation(
    records_path: Path = config.ACCEPTED_RECORDS_PATH,
    tolerance: float = DEFAULT_TOLERANCE,
    report_path: Path = DEFAULT_REPORT_PATH,
    table_path: Path = DEFAULT_TABLE_PATH,
) -> bool:
    """Run all expectations. Returns True when every expectation passes."""
    records_path = Path(records_path)
    if not records_path.is_file():
        raise FileNotFoundError(
            f"Dataset index not found: {records_path}\n"
            "Run `python -m dvc pull` (or `python -m dvc repro`) first."
        )

    if not 0 <= tolerance < 1:
        raise ValueError("--tolerance must be in [0, 1).")

    logger.info("Reading {}", records_path)
    df = pd.read_csv(records_path)

    validator = Validator()

    # Level 1: schema and values. If this fails the proportion tables would be
    # meaningless (or crash), so stop here with the reasons.
    validator.run("records", df, records_expectations())
    if validator.failures:
        _log_failures(validator.failures)
        _write_report(validator, report_path, tolerance, tables=None, table_path=None)
        return False

    # Level 2: proportions.
    tables = build_proportion_tables(df)
    for name, table in tables.items():
        validator.run(name, table, proportion_expectations(tolerance))
        worst = table["abs_deviation"].max()
        logger.info("{}: max deviation {:.4f} (tolerance {:.4f})", name, worst, tolerance)

    _log_deviations(tables, tolerance)
    _write_report(validator, report_path, tolerance, tables, table_path)

    if validator.failures:
        _log_failures(validator.failures)
        logger.error(
            "{} of {} expectations failed.", len(validator.failures), len(validator.results)
        )
        return False

    logger.success("All {} expectations passed.", len(validator.results))
    return True


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def _log_deviations(tables: dict[str, pd.DataFrame], tolerance: float) -> None:
    for name, table in tables.items():
        offenders = table[~(table["abs_deviation"] <= tolerance)]
        if offenders.empty:
            continue
        logger.warning(
            "{}: {} groups above tolerance:\n{}",
            name,
            len(offenders),
            offenders.sort_values("abs_deviation", ascending=False)
            .head(10)
            .to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        )


def _log_failures(failures: list[dict]) -> None:
    for failure in failures:
        logger.error(
            "FAILED [{}] {} {}",
            failure["check"],
            failure["expectation"],
            {k: v for k, v in failure["kwargs"].items() if k not in {"batch_id"}},
        )


def _write_report(
    validator: Validator,
    report_path: Path,
    tolerance: float,
    tables: dict[str, pd.DataFrame] | None,
    table_path: Path | None,
) -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "tolerance": tolerance,
        "success": not validator.failures,
        "n_expectations": len(validator.results),
        "n_failed": len(validator.failures),
        "results": validator.results,
    }
    report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Report saved to {}", report_path)

    if tables is not None and table_path is not None:
        table_path = Path(table_path)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(
            [table.assign(check=name) for name, table in tables.items()],
            ignore_index=True,
        ).to_csv(table_path, index=False)
        logger.info("Proportion tables saved to {}", table_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@app.command()
def main(
    records_path: Annotated[
        Path,
        typer.Option("--records-path", help="Dataset index to validate."),
    ] = config.ACCEPTED_RECORDS_PATH,
    tolerance: Annotated[
        float,
        typer.Option(
            "--tolerance",
            "-t",
            help="Max absolute difference between proportions (0.01 = 1 percentage point).",
        ),
    ] = DEFAULT_TOLERANCE,
    report_path: Annotated[
        Path, typer.Option("--report-path", help="Destination of the JSON report.")
    ] = DEFAULT_REPORT_PATH,
    table_path: Annotated[
        Path, typer.Option("--table-path", help="Destination of the proportions CSV.")
    ] = DEFAULT_TABLE_PATH,
):
    """Validate the class proportions of accepted_records.csv."""
    ok = run_validation(records_path, tolerance, report_path, table_path)
    if not ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
