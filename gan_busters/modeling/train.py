"""
Model training utilities and final-training workflow for GAN-Busters.

This module provides reusable model training functionality with carbon tracking
and MLflow configuration recovery.
"""

import ast

from codecarbon import EmissionsTracker
from loguru import logger
import mlflow
import pandas as pd
from tensorflow import keras

from gan_busters import config
from gan_busters.modeling.architecture import build_model
from gan_busters.modeling.preprocessing import build_dataset
from gan_busters.modeling import tracking


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
    validation_records: pd.DataFrame | None = None,
    carbon_project_name: str = config.CODECARBON_EXPERIMENT_PROJECT,
) -> tuple[keras.callbacks.History, float]:
    """
    Compile and train a model on the supplied training records.
    """
    if optimizer.lower() == "adam":
        optimizer_instance = keras.optimizers.Adam(
            learning_rate=learning_rate
        )

    elif optimizer.lower() == "rmsprop":
        optimizer_instance = keras.optimizers.RMSprop(
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

    validation_dataset = None
    callbacks = []
    
    if validation_records is not None:
        validation_dataset = build_dataset(
            records=validation_records,
            batch_size=config.DEFAULT_EVAL_BATCH_SIZE,
            shuffle=False,
        )

        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            )
        )

    tracker = EmissionsTracker(
        project_name=carbon_project_name,
        save_to_file=False,
    )

    tracker.start()

    try:
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

    finally:
        emissions = tracker.stop()

    logger.info(f"Emissions recorded: {emissions} kg CO2")
    return history, emissions


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
    run_id: str,
    epochs: int,
    model_name: str = config.DEFAULT_MODEL_NAME,
):
    """
    Rebuild a selected experiment configuration, train it on the complete
    development set (training and validation records) with carbon tracking,
    and save the resulting model.
    """

    keras.utils.set_random_seed(config.RANDOM_SEED)
    logger.info("Initializing DagsHub and MLflow tracking...")
    tracking.initialize_mlflow()
    tracking.set_experiment(config.FINAL_MLFLOW_EXPERIMENT)

    logger.info(f"Retrieving hyperparameters from MLflow run ID: {run_id}...")
    cfg = load_run_configuration(run_id)

    df = pd.read_csv(config.ACCEPTED_RECORDS_PATH)
    full_train_records = df[df["split"].isin(["train", "val"])].copy()

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

    with tracking.start_run(config.FINAL_MLFLOW_RUN_NAME):

        final_params = {
            **cfg,
            "source_run_id": run_id,
            "source_max_epochs": cfg["epochs"],
            "final_epochs": epochs,
            "training_records": len(full_train_records),
        }
        final_params.pop("epochs")

        tracking.log_params(final_params)

        _, emissions = train_model(
            model=model,
            train_records=full_train_records,
            optimizer=cfg["optimizer"],
            loss=cfg["loss"],
            learning_rate=cfg["learning_rate"],
            epochs=epochs,
            batch_size=cfg["batch_size"],
            carbon_project_name=config.CODECARBON_FINAL_TRAINING_PROJECT,
        )

        tracking.log_metrics({
            "emissions_kg_co2": emissions,
        })

        config.MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        out_path = config.MODELS_DIR / model_name
        model.save(out_path)

        tracking.log_model(
            model=model,
            name="final_model",
        )

    logger.info(
        f"Training final model on {len(full_train_records)} total training images "
        f"for {epochs} epochs..."
    )
