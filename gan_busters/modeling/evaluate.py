from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import tensorflow as tf

from gan_busters import config
from gan_busters.modeling.preprocessing import build_dataset


def evaluate_model(model: tf.keras.Model, evaluation_records: pd.DataFrame, batch_size: int = config.DEFAULT_BATCH_SIZE) -> dict:
    """
    Reusable evaluation function. Computes loss, Macro F1, ROC-AUC,
    and class-specific precision and recall (decision threshold = 0.5).
    """
    dataset = build_dataset(evaluation_records, batch_size=batch_size, shuffle=False)

    y_true = []
    y_probs = []

    for x_batch, y_batch in dataset:
        probs = model.predict(x_batch, verbose=0)
        y_probs.extend(probs.flatten())
        y_true.extend(y_batch.numpy().flatten())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= 0.5).astype(int)

    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, y_probs)),
        "precision_real": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_real": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    }
    return metrics


def evaluate_final_model():
    """
    Final evaluation workflow: loads the saved final model and computes metrics on test records.
    """
    model_path = config.MODELS_DIR / "gan_busters.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the final model first.")

    logger.info(f"Loading final model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    accepted_records_path = config.PROCESSED_DATA_DIR / "accepted_records.csv"
    df = pd.read_csv(accepted_records_path)
    test_records = df[df["split"] == "test"]

    logger.info(f"Evaluating final model on {len(test_records)} test samples...")
    results = evaluate_model(model, test_records)

    logger.info("Final Test Set Results:")
    for k, v in results.items():
        logger.info(f"{k}: {v:.4f}")
    return results