"""
MLflow experiment tracking utilities for GAN-Busters.

This module configures the connection to the project's DagsHub MLflow
tracking server and provides reusable functions for experiment tracking.
"""

from pathlib import Path

import dagshub
import mlflow
from loguru import logger

from gan_busters import config


def initialize_mlflow() -> None:
    """Configure MLflow to use the project's DagsHub tracking server."""
    logger.info("Initializing MLflow tracking...")
    mlflow.set_tracking_uri(
        f"https://dagshub.com/{config.DAGSHUB_OWNER}/{config.DAGSHUB_REPO}.mlflow"
    )


def set_experiment(experiment_name: str) -> None:
    """Set the active MLflow experiment."""
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to: '{experiment_name}'")


def start_run(run_name: str):
    """Start a named MLflow run."""
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict) -> None:
    """Log the resolved configuration for the active run."""
    if params:
        mlflow.log_params(params)


def log_metrics(metrics: dict) -> None:
    """Log evaluation metrics for the active run."""
    if metrics:
        mlflow.log_metrics(metrics)


def log_history(history) -> None:
    """Log Keras training history as epoch-level MLflow metrics."""
    for metric_name, values in history.history.items():
        for epoch, value in enumerate(values, start=1):
            mlflow.log_metric(
                key=metric_name,
                value=float(value),
                step=epoch,
            )


def log_artifact(
    path: Path | str,
    artifact_path: str | None = None,
) -> None:
    """Log a file as an artifact of the active run if it exists."""
    local_path = Path(path)

    if local_path.exists():
        mlflow.log_artifact(
            local_path=str(local_path),
            artifact_path=artifact_path,
        )


def log_model(model, name: str = "model"):
    """Log a trained TensorFlow/Keras model."""
    return mlflow.keras.log_model(
        model=model,
        name=config.FINAL_LOGGED_MODEL_NAME,
    )