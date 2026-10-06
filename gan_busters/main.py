"""
Central command-line interface for GAN-Busters.

This module provides a single entry point for the project's main workflows,
including model experimentation, final training, and evaluation.

Usage:
    python -m gan_busters.main --help
"""

import typer
from loguru import logger
import tensorflow as tf

from gan_busters import config
from gan_busters.modeling.evaluate import evaluate_final_model
from gan_busters.modeling.experiment import run_experiment
from gan_busters.modeling.predict import predict_images
from gan_busters.modeling.train import train_final_model

app = typer.Typer(
    help="GAN-Busters command-line interface."
)

@app.command("experiment", help="Run a model-selection experiment.")
def experiment(
    epochs: int = typer.Option(config.DEFAULT_EPOCHS, help="Number of epochs"),
    batch_size: int = typer.Option(config.DEFAULT_BATCH_SIZE, help="Batch size"),
    learning_rate: float = typer.Option(config.DEFAULT_LEARNING_RATE, help="Learning rate"),
    kernel_size: int = typer.Option(config.DEFAULT_KERNEL_SIZE, help="Kernel size"),
    padding: str = typer.Option(config.DEFAULT_PADDING, help="Padding"),
    pooling: str = typer.Option(config.DEFAULT_POOLING, help="Pooling"),
    dropout: float = typer.Option(config.DEFAULT_DROPOUT, help="Dropout")
):
    class Args:
        pass
    
    args = Args()
    args.epochs = epochs
    args.batch_size = batch_size
    args.learning_rate = learning_rate
    args.kernel_size = kernel_size
    args.padding = padding
    args.pooling = pooling
    args.dropout = dropout

    run_experiment(args)

@app.command("train", help="Train final model on complete train set using MLflow run ID")
def train(
    run_id: str = typer.Option(..., "--run-id", help="Selected MLflow run ID")
):
    train_final_model(run_id)

@app.command("evaluate", help="Evaluate final model on test split")
def evaluate():
    evaluate_final_model()

@app.command("predict", help="Predict on a single image")
def predict(
    image_path: str = typer.Option(..., "--image-path", help="Path to the image file")
):
    img_bytes = tf.io.read_file(image_path)
    img = tf.io.decode_image(img_bytes, expand_animations=False)
    preds = predict_images(img)
    logger.info(f"Prediction for {image_path}: {preds}")


if __name__ == "__main__":
    app()