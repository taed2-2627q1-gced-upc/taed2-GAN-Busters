"""Register the evaluated final model in the MLflow Model Registry."""

from loguru import logger
import mlflow

from gan_busters import config
from gan_busters.modeling import tracking


def register_model(run_id: str) -> None:
    """Register the final logged model in the MLflow Model Registry."""

    tracking.initialize_mlflow()

    run = mlflow.get_run(run_id)

    logged_models = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        output_format="list",
    )

    final_models = [
        model
        for model in logged_models
        if (
            model.name == config.FINAL_LOGGED_MODEL_NAME
            and model.source_run_id == run_id
        )
    ]

    if not final_models:
        raise ValueError(
            f"No logged model named "
            f"'{config.FINAL_LOGGED_MODEL_NAME}' "
            f"found for run {run_id}."
        )

    logged_model = final_models[0]

    logger.info(
        f"Registering logged model {logged_model.model_id} "
        f"as '{config.REGISTERED_MODEL_NAME}'..."
    )

    model_version = mlflow.register_model(
        model_uri=f"models:/{logged_model.model_id}",
        name=config.REGISTERED_MODEL_NAME,
    )

    logger.success(
        f"Model registered as '{config.REGISTERED_MODEL_NAME}' "
        f"version {model_version.version}."
    )