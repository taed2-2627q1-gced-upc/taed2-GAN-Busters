
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gan_busters import config
from gan_busters.modeling import register


@pytest.fixture
def register_setup(monkeypatch):
    run_id = "test_run_123"

    run = SimpleNamespace(
        info=SimpleNamespace(experiment_id="experiment_456")
    )

    logged_model = SimpleNamespace(
        name=config.FINAL_LOGGED_MODEL_NAME,
        source_run_id=run_id,
        model_id="model_789",
    )

    model_version = SimpleNamespace(version="1")

    mock_get_run = MagicMock(return_value=run)
    mock_search = MagicMock(return_value=[logged_model])
    mock_register = MagicMock(return_value=model_version)
    mock_initialize = MagicMock()

    monkeypatch.setattr(register.mlflow, "get_run", mock_get_run)
    monkeypatch.setattr(
        register.mlflow, "search_logged_models", mock_search
    )
    monkeypatch.setattr(
        register.mlflow, "register_model", mock_register
    )
    monkeypatch.setattr(
        register.tracking, "initialize_mlflow", mock_initialize
    )

    return {
        "run_id": run_id,
        "logged_model": logged_model,
        "get_run": mock_get_run,
        "search": mock_search,
        "register": mock_register,
        "initialize": mock_initialize,
    }


def test_register_model(register_setup):
    mocks = register_setup

    register.register_model(mocks["run_id"])

    mocks["initialize"].assert_called_once()
    mocks["get_run"].assert_called_once_with(mocks["run_id"])

    mocks["search"].assert_called_once_with(
        experiment_ids=["experiment_456"],
        output_format="list",
    )

    mocks["register"].assert_called_once_with(
        model_uri="models:/model_789",
        name=config.REGISTERED_MODEL_NAME,
    )


# Rejects missing, incorrectly named, or incorrectly associated models
@pytest.mark.parametrize(
    "logged_models",
    [
        [],
        [
            SimpleNamespace(
                name="another_model",
                source_run_id="test_run_123",
                model_id="model_1",
            )
        ],
        [
            SimpleNamespace(
                name=config.FINAL_LOGGED_MODEL_NAME,
                source_run_id="different_run",
                model_id="model_2",
            )
        ],
    ],
)
def test_register_model_not_found(register_setup, logged_models):
    mocks = register_setup
    mocks["search"].return_value = logged_models

    with pytest.raises(ValueError, match="No logged model named"):
        register.register_model(mocks["run_id"])

    mocks["register"].assert_not_called() # since the run ID and the model name dont match



# Selects the right model among multiple results
def test_register_model_selects_correct_model(register_setup):
    mocks = register_setup

    mocks["search"].return_value = [
        SimpleNamespace(
            name=config.FINAL_LOGGED_MODEL_NAME,
            source_run_id="another_run",
            model_id="wrong_run_model",
        ),
        SimpleNamespace(
            name="other_model",
            source_run_id=mocks["run_id"],
            model_id="wrong_name_model",
        ),
        mocks["logged_model"],
    ]

    register.register_model(mocks["run_id"])

    mocks["register"].assert_called_once_with(
        model_uri=f"models:/{mocks['logged_model'].model_id}",
        name=config.REGISTERED_MODEL_NAME,
    )
