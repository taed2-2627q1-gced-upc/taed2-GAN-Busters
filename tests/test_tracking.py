
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gan_busters import config
from gan_busters.modeling import tracking


def test_initialize_mlflow(monkeypatch):
    mock_set_uri = MagicMock()
    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", mock_set_uri)

    tracking.initialize_mlflow()

    expected_uri = (
        f"https://dagshub.com/{config.DAGSHUB_OWNER}/"
        f"{config.DAGSHUB_REPO}.mlflow"
    )
    mock_set_uri.assert_called_once_with(expected_uri)


def test_set_experiment(monkeypatch):
    mock_set_experiment = MagicMock()
    monkeypatch.setattr(
        tracking.mlflow, "set_experiment", mock_set_experiment
    )

    tracking.set_experiment("test_experiment")

    mock_set_experiment.assert_called_once_with("test_experiment")


def test_start_run(monkeypatch):
    mock_run = MagicMock()
    mock_start_run = MagicMock(return_value=mock_run)
    monkeypatch.setattr(tracking.mlflow, "start_run", mock_start_run)

    result = tracking.start_run("test_run")

    mock_start_run.assert_called_once_with(run_name="test_run")
    assert result is mock_run


@pytest.mark.parametrize(
    "params",
    [
        {"epochs": 10, "batch_size": 32},
        {},
    ],
)
def test_log_params(monkeypatch, params):
    mock_log_params = MagicMock()
    monkeypatch.setattr(tracking.mlflow, "log_params", mock_log_params)

    tracking.log_params(params)

    if params:
        mock_log_params.assert_called_once_with(params)
    else:
        mock_log_params.assert_not_called()


@pytest.mark.parametrize(
    "metrics",
    [
        {"macro_f1": 0.9, "roc_auc": 0.95},
        {},
    ],
)
def test_log_metrics(monkeypatch, metrics):
    mock_log_metrics = MagicMock()
    monkeypatch.setattr(tracking.mlflow, "log_metrics", mock_log_metrics)

    tracking.log_metrics(metrics)

    if metrics:
        mock_log_metrics.assert_called_once_with(metrics)
    else:
        mock_log_metrics.assert_not_called()


def test_log_history(monkeypatch):
    mock_log_metric = MagicMock()
    monkeypatch.setattr(tracking.mlflow, "log_metric", mock_log_metric)

    history = SimpleNamespace(history={
        "loss": [0.5, 0.3],
        "accuracy": [0.7, 0.9],
    })

    tracking.log_history(history)

    assert mock_log_metric.call_count == 4

    mock_log_metric.assert_any_call(
        key="loss", value=0.5, step=1
    )
    mock_log_metric.assert_any_call(
        key="loss", value=0.3, step=2
    )
    mock_log_metric.assert_any_call(
        key="accuracy", value=0.7, step=1
    )
    mock_log_metric.assert_any_call(
        key="accuracy", value=0.9, step=2
    )


@pytest.mark.parametrize("artifact_path", [None, "figures"])
def test_log_artifact_existing_file(
    monkeypatch, tmp_path, artifact_path
):
    mock_log_artifact = MagicMock()
    monkeypatch.setattr(
        tracking.mlflow, "log_artifact", mock_log_artifact
    )

    file_path = tmp_path / "plot.png"
    file_path.touch()

    tracking.log_artifact(file_path, artifact_path=artifact_path)

    mock_log_artifact.assert_called_once_with(
        local_path=str(file_path),
        artifact_path=artifact_path,
    )


def test_log_artifact_missing_file(monkeypatch, tmp_path):
    mock_log_artifact = MagicMock()
    monkeypatch.setattr(
        tracking.mlflow, "log_artifact", mock_log_artifact
    )

    tracking.log_artifact(tmp_path / "missing.png")

    mock_log_artifact.assert_not_called()


@pytest.mark.parametrize(
    "name",
    [
        config.FINAL_LOGGED_MODEL_NAME,
        "custom_model",
    ],
)
def test_log_model(monkeypatch, name):
    model = MagicMock()
    mock_result = MagicMock()
    mock_log_model = MagicMock(return_value=mock_result)

    mock_keras = MagicMock()
    mock_keras.log_model = mock_log_model

    monkeypatch.setattr(
        tracking.mlflow,
        "keras",
        mock_keras,
    )

    result = tracking.log_model(model, name=name)

    mock_log_model.assert_called_once_with(
        model=model,
        name=name,
    )
    assert result is mock_result