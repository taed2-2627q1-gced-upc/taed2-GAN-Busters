"""
Central command-line interface for GAN-Busters.

This module provides a single entry point for the project's main workflows,
including model experimentation, final training, evaluation, and prediction.

Usage:
    python -m gan_busters.main --help
"""

from pathlib import Path
import typer
from loguru import logger

from gan_busters import config
from gan_busters.modeling.evaluate import evaluate_final_model
from gan_busters.modeling.experiment import run_experiment
from gan_busters.modeling.predict import predict_from_path
from gan_busters.modeling.train import train_final_model

app = typer.Typer(
    help="GAN-Busters command-line interface."
)

@app.command("experiment", help="Run a model-selection experiment.")
def experiment(
    experiment_name: str = typer.Option(
    config.DEFAULT_MLFLOW_EXPERIMENT,
    "--experiment-name",
    help="MLflow experiment name",
    ),
    run_name: str = typer.Option(
        ...,
        "--run-name",
        help="Name assigned to the MLflow run.",
    ),
    epochs: int = typer.Option(
        config.DEFAULT_EPOCHS,
        help="Number of epochs",
    ),
    batch_size: int = typer.Option(
        config.DEFAULT_BATCH_SIZE,
        help="Batch size",
    ),
    learning_rate: float = typer.Option(
        config.DEFAULT_LEARNING_RATE,
        help="Learning rate",
    ),
    kernel_size: int = typer.Option(
        config.DEFAULT_KERNEL_SIZE,
        help="Kernel size",
    ),
    padding: str = typer.Option(
        config.DEFAULT_PADDING,
        help="Padding",
    ),
    pooling: str = typer.Option(
        config.DEFAULT_POOLING,
        help="Pooling",
    ),
    dropout: float = typer.Option(
        config.DEFAULT_DROPOUT,
        help="Dropout",
    ),
):
    run_experiment(
        run_name=run_name,
        experiment_name=experiment_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        kernel_size=kernel_size,
        padding=padding,
        pooling=pooling,
        dropout=dropout,
)

@app.command(
    "train",
    help="Train final model on the complete train set using MLflow run ID",
)
def train(
    run_id: str = typer.Option(
        ...,
        "--run-id",
        help="Selected MLflow run ID",
    ),
    model_name: str = typer.Option(
        config.DEFAULT_MODEL_NAME,
        "--model-name",
        help="Filename for the final trained model",
    ),
):
    train_final_model(
        run_id=run_id,
        model_name=model_name,
    )

@app.command("evaluate", help="Evaluate final model on test split")
def evaluate(
    model_name: str = typer.Option(
        config.DEFAULT_MODEL_NAME,
        "--model-name",
        help="Saved model filename inside the models directory.",
    ),
    batch_size: int = typer.Option(
        config.DEFAULT_EVAL_BATCH_SIZE,
        "--batch-size",
        help="Batch size used during evaluation.",
    ),
    threshold: float = typer.Option(
        config.DEFAULT_THRESHOLD,
        "--threshold",
        help="Probability threshold used to classify an image as FAKE.",
    ),
):
    evaluate_final_model(
        model_name=model_name,
        batch_size=batch_size,
        threshold=threshold,
    )


@app.command("predict", help="Predict on one image or a directory of images")
def predict(
    path: Path = typer.Option(
        ...,
        "--path",
        help="Path to an image file or a directory containing images.",
        exists=True,
    ),
    model_name: str = typer.Option(
        config.DEFAULT_MODEL_NAME,
        "--model-name",
        help="Saved model filename inside the models directory.",
    ),
    threshold: float = typer.Option(
        config.DEFAULT_THRESHOLD,
        "--threshold",
        help="Probability threshold used to classify an image as FAKE.",
    ),
    output_file: str = typer.Option(
        config.DEFAULT_PREDICTIONS_FILE,
        "--output-file",
        help="Name of the CSV file used to save predictions.",
    ),
):
    predictions = predict_from_path(
        path=path,
        model_name=model_name,
        threshold=threshold,
        output_file=output_file,
    )

    logger.info(
        f"Saved {len(predictions)} predictions to "
        f"{config.PREDICTIONS_DIR / output_file}"
    )


if __name__ == "__main__":
    app()