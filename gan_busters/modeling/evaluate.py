"""
Model evaluation utilities and final-evaluation workflow for GAN-Busters.

This module provides reusable functions for evaluating trained models on
labelled image records and a workflow for evaluating the final saved model
on the predefined CIFAKE test set.
"""

import pandas as pd
from loguru import logger

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow import keras

from gan_busters import config
from gan_busters.modeling.preprocessing import build_dataset
from gan_busters.modeling import tracking


# -------------------------------------------------------------------------
# Reusable evaluation function
# -------------------------------------------------------------------------

def evaluate_model(
    model: keras.Model,
    evaluation_records: pd.DataFrame,
    threshold: float = config.DEFAULT_THRESHOLD,
) -> dict:
    """
    Evaluate a trained model on labelled image records.

    The model output is interpreted as P(FAKE), where REAL = 0 and FAKE = 1.

    Returns:
        Dictionary containing loss, Macro F1, ROC-AUC, and class-specific
        F1, precision, and recall.
    """

    dataset = build_dataset(
        records=evaluation_records,
        batch_size=config.DEFAULT_EVAL_BATCH_SIZE,
        shuffle=False,
    )

    # Ground-truth labels: REAL = 0, FAKE = 1.
    y_true = (
        evaluation_records["label"]
        .map(config.LABEL_MAPPING)
        .astype(int)
        .to_numpy()
    )

    # Predict P(FAKE).
    probabilities = model.predict(
        dataset,
        verbose=0,
    ).reshape(-1)

    # Convert probabilities to binary predictions.
    predictions = (probabilities >= threshold).astype(int)

    # Evaluate the model loss using the configured Keras loss.
    loss = model.evaluate(
        dataset,
        verbose=0,
    )

    metrics = {
        "loss": float(loss),

        # Overall metrics
        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),

        # REAL = 0
        "real_f1": f1_score(
            y_true,
            predictions,
            pos_label=0,
            zero_division=0,
        ),
        "real_precision": precision_score(
            y_true,
            predictions,
            pos_label=0,
            zero_division=0,
        ),
        "real_recall": recall_score(
            y_true,
            predictions,
            pos_label=0,
            zero_division=0,
        ),

        # FAKE = 1
        "fake_f1": f1_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "fake_precision": precision_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "fake_recall": recall_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
    }

    return metrics


# -------------------------------------------------------------------------
# Final-evaluation workflow
# -------------------------------------------------------------------------

def evaluate_final_model(
    model_name: str = config.DEFAULT_MODEL_NAME,
    threshold: float = config.DEFAULT_THRESHOLD,
) -> dict:
    """Evaluate a saved model on the predefined CIFAKE test set."""

    model_path = config.MODELS_DIR / model_name

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {model_path}"
        )

    # Load the complete model, including embedded preprocessing.
    model = keras.models.load_model(model_path)

    # Load accepted records and isolate the untouched CIFAKE test set.
    records = pd.read_csv(config.ACCEPTED_RECORDS_PATH)

    test_records = records[
        records["split"] == "test"
    ].copy()

    metrics = evaluate_model(
        model=model,
        evaluation_records=test_records,
        threshold=threshold,
    )

    metrics["test_loss"] = metrics.pop("loss")

    # Log final evaluation to MLflow.
    tracking.initialize_mlflow()
    tracking.set_experiment(config.FINAL_EVALUATION_MLFLOW_EXPERIMENT)

    with tracking.start_run(config.FINAL_EVALUATION_MLFLOW_RUN_NAME):
        tracking.log_params({
            "model_name": model_name,
            "threshold": threshold,
            "test_records": len(test_records),
        })

        tracking.log_metrics(metrics)

    logger.success(
        f"Final CIFAKE test evaluation completed. "
        f"Macro F1: {metrics['macro_f1']:.4f}, "
        f"ROC-AUC: {metrics['roc_auc']:.4f}"
    )

    return metrics