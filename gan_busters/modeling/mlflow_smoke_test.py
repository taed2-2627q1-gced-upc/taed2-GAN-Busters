"""
Smoke test for the GAN-Busters MLflow/DagsHub connection.

This script creates a small test experiment and logs one parameter
and one metric. It does not train or save a model.

Usage:
    python -m gan_busters.modeling.smoke_test
"""

import mlflow

from gan_busters.modeling.tracking import (
    initialize_mlflow,
    set_experiment,
    start_run,
    log_params,
    log_metrics,
)


def main():
    initialize_mlflow()

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    set_experiment("smoke-test")

    with start_run("connection-test"):
        log_params({
            "test_parameter": 123,
            "purpose": "dagshub_connection_test",
        })

        log_metrics({
            "test_metric": 0.99,
        })

    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()