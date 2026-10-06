from loguru import logger
import mlflow

from gan_busters.modeling.tracking import init_tracking

'''
Explicit test to verify that DagsHub and MLflow are correctly 
configured and can log parameters and metrics.
'''
def run_smoke_test():
    logger.info("Executing smoke test for DagsHub / MLflow connection...")
    init_tracking(experiment_name="smoke-test")

    with mlflow.start_run(run_name="connection-test"):
        mlflow.log_param("test_parameter", 123)
        mlflow.log_metric("test_metric", 0.99)

    logger.success("Smoke test passed! Check your DagsHub repository under 'Experiments'.")


if __name__ == "__main__":
    run_smoke_test()