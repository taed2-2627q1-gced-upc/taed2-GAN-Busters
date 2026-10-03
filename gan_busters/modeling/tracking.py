"""
MLflow experiment tracking utilities for GAN-Busters.

This module configures the connection to the project's DagsHub MLflow
tracking server and provides reusable functions for experiment tracking.
"""

from pathlib import Path

import dagshub
import mlflow

from gan_busters.config import DAGSHUB_OWNER, DAGSHUB_REPO


def initialize_mlflow() -> None:
    """Configure MLflow to use the project's DagsHub tracking server."""

    dagshub.init(
        repo_owner=DAGSHUB_OWNER,
        repo_name=DAGSHUB_REPO,
        mlflow=True,
    )


def set_experiment(experiment_name: str) -> None:
    """Set the active MLflow experiment."""

    mlflow.set_experiment(experiment_name)


def start_run(run_name: str):
    """Start a named MLflow run."""

    return mlflow.start_run(run_name=run_name)


def log_params(params: dict) -> None:
    """Log the resolved configuration for the active run."""

    mlflow.log_params(params)


def log_metrics(metrics: dict) -> None:
    """Log evaluation metrics for the active run."""

    mlflow.log_metrics(metrics)


def log_artifact(
    path: Path,
    artifact_path: str | None = None,
) -> None:
    """Log a file as an artifact of the active run."""

    mlflow.log_artifact(
        local_path=str(path),
        artifact_path=artifact_path,
    )


def log_model(model, name: str = "model") -> None:
    """Log a trained TensorFlow/Keras model."""

    mlflow.keras.log_model(
        model=model,
        name=name,
    )