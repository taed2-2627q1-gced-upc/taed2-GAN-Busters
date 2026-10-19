"""Evaluate the registered GAN-Busters model on external datasets."""

import mlflow
import pandas as pd
from loguru import logger

from gan_busters import config
from gan_busters.modeling import tracking
from gan_busters.modeling.evaluate import evaluate_model


def evaluate_external_model(
    model_version: int = config.ROBUSTNESS_MODEL_VERSION,
    threshold: float = config.DEFAULT_THRESHOLD,
    batch_size: int = config.DEFAULT_EVAL_BATCH_SIZE,
) -> dict:
    """Evaluate the registered model independently on each generator."""

    tracking.initialize_mlflow()
    tracking.set_experiment(config.EXTERNAL_EVALUATION_MLFLOW_EXPERIMENT)

    model_uri = f"models:/{config.REGISTERED_MODEL_NAME}/{model_version}"
    model = mlflow.keras.load_model(model_uri)

    results = {}

    for generator in config.EXTERNAL_GENERATORS:
        records_path = (
            config.EXTERNAL_PROCESSED_DIR
            / generator
            / "accepted_records.csv"
        )

        records = pd.read_csv(records_path)

        test_records = records[records["split"] == "test"].copy()

        metrics = evaluate_model(
            model=model,
            evaluation_records=test_records,
            threshold=threshold,
            batch_size=batch_size,
            image_dir=config.EXTERNAL_RAW_DIR / generator,
            resize_before_batch=True,
        )

        with tracking.start_run(generator):
            tracking.log_params({
                "dataset": "Tiny GenImage",
                "generator": generator,
                "model_uri": model_uri,
                "threshold": threshold,
                "batch_size": batch_size,
                "test_records": len(test_records),
                "resize_before_batch": True,
            })

            tracking.log_metrics(metrics)

        results[generator] = metrics

        logger.success(
            f"{generator}: "
            f"Macro F1 = {metrics['macro_f1']:.4f}, "
            f"ROC-AUC = {metrics['roc_auc']:.4f}"
        )

    return results