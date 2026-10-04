"""
Model training utilities and final-training workflow for GAN-Busters.

This module provides reusable model training functionality with carbon tracking
and MLflow configuration recovery.
"""

import ast
from pathlib import Path

from codecarbon import EmissionsTracker
from loguru import logger
import mlflow
import pandas as pd
from tensorflow import keras
import typer

from gan_busters.config import PROCESSED_DATA_DIR, MODELS_DIR
from gan_busters.modeling.architecture import build_model
from gan_busters.modeling.preprocessing import build_dataset
from gan_busters.modeling.tracking import initialize_mlflow


# -------------------------------------------------------------------------
# Reusable training function
# -------------------------------------------------------------------------

def train_model(
    model: keras.Model,
    train_records: pd.DataFrame,
    optimizer: str,
    loss: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
) -> keras.callbacks.History:
    """
    Compile and train a model on the supplied training records.
    """
    if optimizer.lower() == "adam":
        optimizer_instance = keras.optimizers.Adam(
            learning_rate=learning_rate
        )
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer}"
        )

    model.compile(
        optimizer=optimizer_instance,
        loss=loss,
    )

    train_dataset = build_dataset(
        records=train_records,
        batch_size=batch_size,
        shuffle=True,
    )

    history = model.fit(
        train_dataset,
        epochs=epochs,
        verbose=1,
    )

    return history


# -------------------------------------------------------------------------
# MLflow configuration recovery
# -------------------------------------------------------------------------

def load_run_configuration(run_id: str) -> dict:
    """
    Retrieve and convert the configuration stored in an MLflow run.

    MLflow parameters are stored as strings, so values are converted back
    to the Python types expected by the model and training functions.
    """
    run = mlflow.get_run(run_id)
    params = run.data.params

    return {
        "input_shape": ast.literal_eval(params["input_shape"]),
        "conv_filters": ast.literal_eval(params["conv_filters"]),
        "kernel_size": int(params["kernel_size"]),
        "padding": params["padding"],
        "conv_strides": ast.literal_eval(params["conv_strides"]),
        "pooling": params["pooling"],
        "pool_size": ast.literal_eval(params["pool_size"]),
        "dense_units": ast.literal_eval(params["dense_units"]),
        "dropout": float(params["dropout"]),
        "optimizer": params["optimizer"],
        "loss": params["loss"],
        "learning_rate": float(params["learning_rate"]),
        "batch_size": int(params["batch_size"]),
        "epochs": int(params["epochs"]),
    }


# -------------------------------------------------------------------------
# Final-training workflow
# -------------------------------------------------------------------------

def train_final_model(
    run_id: str = typer.Option(
        ...,
        help="MLflow run ID containing the selected model configuration.",
    ),
    model_name: str = typer.Option(
        "gan_busters.keras",
        help="Filename for the final trained model.",
    ),
):
    """
    Rebuild a selected experiment configuration, train it on the complete
    predefined training set with carbon tracking, and save the resulting model.
    """
    logger.info("Initializing DagsHub and MLflow tracking...")
    initialize_mlflow()

    logger.info(f"Retrieving hyperparameters from MLflow run ID: {run_id}...")
    cfg = load_run_configuration(run_id)

    accepted_records_path = PROCESSED_DATA_DIR / "accepted_records.csv"
    df = pd.read_csv(accepted_records_path)
    full_train_records = df[df["split"] == "train"].copy()

    logger.info(f"Building model from configuration for run {run_id}...")
    model = build_model(
        input_shape=cfg["input_shape"],
        conv_filters=cfg["conv_filters"],
        kernel_size=cfg["kernel_size"],
        padding=cfg["padding"],
        conv_strides=cfg["conv_strides"],
        pooling=cfg["pooling"],
        pool_size=cfg["pool_size"],
        dense_units=cfg["dense_units"],
        dropout=cfg["dropout"],
    )

    logger.info(f"Training final model on {len(full_train_records)} total training images...")
    tracker = EmissionsTracker(project_name="Final_Model_Training", save_to_file=False)
    tracker.start()

    train_model(
        model=model,
        train_records=full_train_records,
        optimizer=cfg["optimizer"],
        loss=cfg["loss"],
        learning_rate=cfg["learning_rate"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
    )

    emissions = tracker.stop()
    logger.info(f"Emissions recorded: {emissions} kg CO2")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / model_name
    model.save(out_path)
    logger.success(f"Final model saved to {out_path}")