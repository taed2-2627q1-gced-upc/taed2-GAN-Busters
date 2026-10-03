import os

import dagshub
from loguru import logger
import mlflow

# Constant project references
MLFLOW_EXPERIMENT_NAME = "CIFAKE_Experiments"
DAGSHUB_REPO_OWNER = "taed2-2627q1-gced-upc"
DAGSHUB_REPO_NAME = "taed2-GAN-Busters"


def init_tracking(experiment_name: str = MLFLOW_EXPERIMENT_NAME):
    """
    Initializes DagsHub integration and sets the active MLflow experiment.
    """
    logger.info("Initializing DagsHub and MLflow tracking...")
    dagshub.init(
        repo_owner=DAGSHUB_REPO_OWNER,
        repo_name=DAGSHUB_REPO_NAME,
        mlflow=True
    )
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to: '{experiment_name}'")


def log_run_data(params: dict | None = None, metrics: dict | None = None, artifacts: list | None = None):
    """
    Logs parameters, metrics, and local artifact files to the active MLflow run.
    """
    if params:
        mlflow.log_params(params)
    if metrics:
        mlflow.log_metrics(metrics)
    if artifacts:
        for path in artifacts:
            if os.path.exists(path):
                mlflow.log_artifact(str(path))