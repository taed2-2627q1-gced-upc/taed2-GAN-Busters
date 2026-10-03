"""
Central command-line interface for GAN-Busters.

This module provides a single entry point for the project's main workflows,
including model experimentation, final training, and evaluation.

Usage:
    python -m gan_busters.main --help
"""

import typer

from gan_busters.modeling.experiment import run_experiment
from gan_busters.modeling.train import train_final_model
from gan_busters.modeling.evaluate import evaluate_final_model


app = typer.Typer(
    help="GAN-Busters command-line interface."
)


app.command(
    "experiment",
    help="Run a model-selection experiment.",
)(run_experiment)

app.command(
    "train",
    help="Train a final model from a selected MLflow run.",
)(train_final_model)

app.command(
    "evaluate",
    help="Evaluate a saved model on the predefined test set.",
)(evaluate_final_model)


if __name__ == "__main__":
    app()