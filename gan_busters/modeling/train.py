"""
Model training utilities and final-training workflow for GAN-Busters.

This module provides reusable model training functionality.

When executed directly, the selected configuration is retrieved from an
MLflow experiment run, rebuilt as a fresh model, trained on the complete
predefined training set, and saved as the final trained model.
"""

import ast
import typer

import mlflow
import pandas as pd
from tensorflow import keras

from gan_busters.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
)
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

    This function does not save the model and does not interact with MLflow.
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
    predefined training set, and save the resulting model.
    """

    # Connect to the project's MLflow tracking server.
    initialize_mlflow()

    # Retrieve the selected experiment configuration.
    config = load_run_configuration(run_id)

    # Load all accepted records belonging to the predefined training set.
    records = pd.read_csv(
        PROCESSED_DATA_DIR / "accepted_records.csv"
    )

    train_records = records[
        records["split"] == "train"
    ].copy()

    # Build a fresh, untrained model from the selected configuration.
    model = build_model(
        input_shape=config["input_shape"],
        conv_filters=config["conv_filters"],
        kernel_size=config["kernel_size"],
        padding=config["padding"],
        conv_strides=config["conv_strides"],
        pooling=config["pooling"],
        pool_size=config["pool_size"],
        dense_units=config["dense_units"],
        dropout=config["dropout"],
    )

    # Train from scratch using the complete predefined training set.
    train_model(
        model=model,
        train_records=train_records,
        optimizer=config["optimizer"],
        loss=config["loss"],
        learning_rate=config["learning_rate"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
    )

    # Save the final trained model.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / model_name
    model.save(model_path)

    print(f"Final model saved to: {model_path}")
    print(f"Configuration selected from MLflow run: {run_id}")


if __name__ == "__main__":
    app()
