"""
Model-selection experiment runner for GAN-Busters.

This module orchestrates a single training and validation experiment.

Configuration defaults are loaded from config.py and may be overridden
through CLI arguments. Each run builds a fresh model, trains it on the
training subset, evaluates it on the validation subset, and logs the
configuration and results to MLflow/DagsHub.

The predefined CIFAKE test set is never used by this workflow.
"""

from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

from gan_busters import config

from gan_busters.modeling.architecture import build_model
from gan_busters.modeling.train import train_model
from gan_busters.modeling.evaluate import evaluate_model
from gan_busters.modeling import tracking


def run_experiment(
    run_name: str,
    experiment_name: str = config.DEFAULT_MLFLOW_EXPERIMENT,
    kernel_size: int = config.DEFAULT_KERNEL_SIZE,
    padding: str = config.DEFAULT_PADDING,
    pooling: str = config.DEFAULT_POOLING,
    learning_rate: float = config.DEFAULT_LEARNING_RATE,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    dropout: float = config.DEFAULT_DROPOUT,
    epochs: int = config.DEFAULT_EPOCHS,
):
    """Run one model-selection experiment."""
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        logger.info(f"GPU available: {gpus}")
    else:
        logger.info("No GPU detected. Training will use CPU.")

    # ---------------------------------------------------------
    # Load predefined training records
    # ---------------------------------------------------------
    records = pd.read_csv(config.ACCEPTED_RECORDS_PATH)

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
        stratification = "label_subclass"
    else:
        stratify_by = train_records["label"]
        stratification = "label"

    train_records, validation_records = train_test_split(
        train_records,
        test_size=config.DEFAULT_VALIDATION_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=stratify_by,
    )

    config.VALIDATION_RECORDS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_records.to_csv(
        config.VALIDATION_RECORDS_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Build model
    # ---------------------------------------------------------
    model = build_model(
        input_shape=config.INPUT_SHAPE,
        conv_filters=config.DEFAULT_CONV_FILTERS,
        kernel_size=kernel_size,
        padding=padding,
        conv_strides=config.DEFAULT_CONV_STRIDES,
        pooling=pooling,
        pool_size=config.DEFAULT_POOL_SIZE,
        dense_units=config.DEFAULT_DENSE_UNITS,
        dropout=dropout,
    )

    # ---------------------------------------------------------
    # Configure MLflow
    # ---------------------------------------------------------
    tracking.initialize_mlflow()
    tracking.set_experiment(experiment_name)

    params = {
        "input_shape": str(config.INPUT_SHAPE),
        "conv_filters": str(config.DEFAULT_CONV_FILTERS),
        "kernel_size": kernel_size,
        "padding": padding,
        "conv_strides": str(config.DEFAULT_CONV_STRIDES),
        "pooling": pooling,
        "pool_size": str(config.DEFAULT_POOL_SIZE),
        "dense_units": str(config.DEFAULT_DENSE_UNITS),
        "dropout": dropout,
        "optimizer": config.DEFAULT_OPTIMIZER,
        "loss": config.DEFAULT_LOSS,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "validation_size": config.DEFAULT_VALIDATION_SIZE,
        "random_seed": config.RANDOM_SEED,
        "stratification": stratification,
    }

    # ---------------------------------------------------------
    # Train, track emissions, and evaluate
    # ---------------------------------------------------------
    with tracking.start_run(run_name):
        tracking.log_params(params)

        logger.info(f"Starting run: {run_name}")
        history, emissions = train_model(
            model=model,
            train_records=train_records,
            validation_records=validation_records,
            optimizer=config.DEFAULT_OPTIMIZER,
            loss=config.DEFAULT_LOSS,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
        )

        tracking.log_history(history)
        tracking.log_metrics({"emissions_kg_co2": emissions})

        metrics = evaluate_model(
            model=model,
            evaluation_records=validation_records,
        )

        metrics["validation_loss"] = metrics.pop("loss")

        tracking.log_metrics(metrics)
        logger.success(f"Run completed. Validation Macro F1: {metrics['macro_f1']:.4f}")