"""
Model-selection experiment runner for GAN-Busters.

This module orchestrates a single training and validation experiment.

Configuration defaults are loaded from config.py and may be overridden
through CLI arguments. Each run builds a fresh model, trains it on the
training subset, evaluates it on the validation subset, and logs the
configuration and results to MLflow/DagsHub.

The predefined CIFAKE test set is never used by this workflow.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from loguru import logger


from gan_busters.config import (
    PROCESSED_DATA_DIR,
    INPUT_SHAPE,
    DEFAULT_CONV_FILTERS,
    DEFAULT_DENSE_UNITS,
    DEFAULT_KERNEL_SIZE,
    DEFAULT_PADDING,
    DEFAULT_CONV_STRIDES,
    DEFAULT_POOLING,
    DEFAULT_POOL_SIZE,
    DEFAULT_LOSS,
    DEFAULT_OPTIMIZER,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_VALIDATION_SIZE,
    DEFAULT_MLFLOW_EXPERIMENT,
    RANDOM_SEED,
)

from gan_busters.modeling.architecture import build_model
from gan_busters.modeling.train import train_model
from gan_busters.modeling.evaluate import evaluate_model
from gan_busters.modeling.tracking import (
    initialize_mlflow,
    set_experiment,
    start_run,
    log_params,
    log_metrics,
)


def run_experiment(
    experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT,
    run_name: str = "baseline",
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    padding: str = DEFAULT_PADDING,
    pooling: str = DEFAULT_POOLING,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dropout: float = DEFAULT_DROPOUT,
    epochs: int = DEFAULT_EPOCHS,
):
    """Run one model-selection experiment."""
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        logger.info(f"GPU available: {gpus}")
    else:
        logger.info("No GPU detected. Training will use CPU.")

    # ---------------------------------------------------------
    # Load predefined training records
    # ---------------------------------------------------------

    records = pd.read_csv(
        PROCESSED_DATA_DIR / "accepted_records.csv"
    )

    train_records = records[
        records["split"] == "train"
    ].copy()

    # ---------------------------------------------------------
    # Create reproducible train / validation split
    # ---------------------------------------------------------

    if train_records["subclass"].notna().all():
        stratify_by = (
            train_records["label"].astype(str)
            + "_"
            + train_records["subclass"].astype(str)
        )
    else:
        stratify_by = train_records["label"]

    train_records, validation_records = train_test_split(
        train_records,
        test_size=DEFAULT_VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify_by,
    )

    # ---------------------------------------------------------
    # Build model
    # ---------------------------------------------------------

    model = build_model(
        input_shape=INPUT_SHAPE,
        conv_filters=DEFAULT_CONV_FILTERS,
        kernel_size=kernel_size,
        padding=padding,
        conv_strides=DEFAULT_CONV_STRIDES,
        pooling=pooling,
        pool_size=DEFAULT_POOL_SIZE,
        dense_units=DEFAULT_DENSE_UNITS,
        dropout=dropout,
    )

    # ---------------------------------------------------------
    # Configure MLflow
    # ---------------------------------------------------------

    initialize_mlflow()
    set_experiment(experiment_name)

    params = {
        "input_shape": str(INPUT_SHAPE),
        "conv_filters": str(DEFAULT_CONV_FILTERS),
        "kernel_size": kernel_size,
        "padding": padding,
        "conv_strides": str(DEFAULT_CONV_STRIDES),
        "pooling": pooling,
        "pool_size": str(DEFAULT_POOL_SIZE),
        "dense_units": str(DEFAULT_DENSE_UNITS),
        "dropout": dropout,
        "optimizer": DEFAULT_OPTIMIZER,
        "loss": DEFAULT_LOSS,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "validation_size": DEFAULT_VALIDATION_SIZE,
        "random_seed": RANDOM_SEED,
    }

    # ---------------------------------------------------------
    # Train and evaluate
    # ---------------------------------------------------------

    with start_run(run_name):

        log_params(params)

        history = train_model(
            model=model,
            train_records=train_records,
            optimizer=DEFAULT_OPTIMIZER,
            loss=DEFAULT_LOSS,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
        )

        metrics = evaluate_model(
            model=model,
            evaluation_records=validation_records,
            batch_size=batch_size,
        )

        log_metrics(metrics)