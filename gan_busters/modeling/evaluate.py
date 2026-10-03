"""
Model evaluation utilities and final-evaluation workflow for GAN-Busters.

This module provides reusable functions for evaluating trained models on
labelled image records.

When executed directly, this module loads a saved model and evaluates it on
the predefined CIFAKE test set.
"""

import pandas as pd
import typer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow import keras

from gan_busters.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    DEFAULT_BATCH_SIZE,
)
from gan_busters.modeling.preprocessing import build_dataset



# -------------------------------------------------------------------------
# Reusable evaluation function
# -------------------------------------------------------------------------

def evaluate_model(
    model: keras.Model,
    evaluation_records: pd.DataFrame,
    batch_size: int,
    threshold: float = 0.5,
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
        batch_size=batch_size,
    )

    # Ground-truth labels: REAL = 0, FAKE = 1.
    y_true = (
        evaluation_records["label"]
        .map({"REAL": 0, "FAKE": 1})
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
    model_name: str = typer.Option(
        "gan_busters.keras",
        help="Saved model filename inside the models directory.",
    ),
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE,
        help="Batch size used during evaluation.",
    ),
    threshold: float = typer.Option(
        0.5,
        help="Probability threshold used to classify an image as FAKE.",
    ),
):
    """Evaluate a saved model on the predefined CIFAKE test set."""

    model_path = MODELS_DIR / model_name

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {model_path}"
        )

    # Load the complete model, including embedded preprocessing.
    model = keras.models.load_model(model_path)

    # Load accepted records and isolate the untouched CIFAKE test set.
    records = pd.read_csv(
        PROCESSED_DATA_DIR / "accepted_records.csv"
    )

    test_records = records[
        records["split"] == "test"
    ].copy()

    metrics = evaluate_model(
        model=model,
        evaluation_records=test_records,
        batch_size=batch_size,
        threshold=threshold,
    )

    print("\nCIFAKE test results")
    print("-------------------")
    print(f"threshold: {threshold}")

    for name, value in metrics.items():
        print(f"{name}: {value}")