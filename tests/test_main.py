
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from gan_busters import config
from gan_busters import main


runner = CliRunner()


@pytest.fixture
def cli_setup(monkeypatch):
    mocks = {}

    for function_name in [
        "run_experiment",
        "train_final_model",
        "evaluate_final_model",
        "register_model",
        "predict_from_path",
    ]:
        mock = MagicMock()
        monkeypatch.setattr(main, function_name, mock)
        mocks[function_name] = mock

    return mocks


# verifying that typer converts the cli arguments into the right python types, for the experiment call
def test_experiment_command(cli_setup):
    result = runner.invoke(main.app, [
        "experiment",
        "--run-name", "test_run",
        "--experiment-name", "test_experiment",
        "--epochs", "10",
        "--batch-size", "64",
        "--learning-rate", "0.0001",
        "--kernel-size", "5",
        "--padding", "same",
        "--pooling", "avg",
        "--dropout", "0.25",
    ])

    assert result.exit_code == 0

    cli_setup["run_experiment"].assert_called_once_with(
        run_name="test_run",
        experiment_name="test_experiment",
        epochs=10,
        batch_size=64,
        learning_rate=0.0001,
        kernel_size=5,
        padding="same",
        pooling="avg",
        dropout=0.25,
    )


# same but for training
def test_train_command(cli_setup):
    result = runner.invoke(main.app, [
        "train",
        "--run-id", "test_run_123",
        "--epochs", "20",
        "--model-name", "final.keras",
    ])

    assert result.exit_code == 0

    cli_setup["train_final_model"].assert_called_once_with(
        run_id="test_run_123",
        epochs=20,
        model_name="final.keras",
    )



# evaluation
def test_evaluate_command(cli_setup):
    result = runner.invoke(main.app, [
        "evaluate",
        "--model-name", "final.keras",
        "--batch-size", "16",
        "--threshold", "0.6",
    ])

    assert result.exit_code == 0

    cli_setup["evaluate_final_model"].assert_called_once_with(
        model_name="final.keras",
        threshold=0.6,
        batch_size=16,
    )


# register
def test_register_command(cli_setup):
    result = runner.invoke(main.app, [
        "register",
        "--run-id", "test_run_123",
    ])

    assert result.exit_code == 0

    cli_setup["register_model"].assert_called_once_with(
        run_id="test_run_123",
    )


# predict from a directory path

def test_predict_command(cli_setup, tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.touch()

    cli_setup["predict_from_path"].return_value = [
        {
            "image_path": str(image_path),
            "label": "FAKE",
            "class_probability": 0.8,
        }
    ]

    result = runner.invoke(main.app, [
        "predict",
        "--path", str(image_path),
        "--model-name", "final.keras",
        "--threshold", "0.6",
        "--output-file", "predictions.csv",
    ])

    assert result.exit_code == 0

    cli_setup["predict_from_path"].assert_called_once_with(
        path=image_path,
        model_name="final.keras",
        threshold=0.6,
        output_file="predictions.csv",
    )



### testing invalid CLI arguments ####

@pytest.mark.parametrize(
    "arguments",
    [
        ["experiment"],                    # Missing run name
        ["train", "--epochs", "20"],        # Missing run ID
        ["train", "--run-id", "123"],       # Missing epochs
        ["register"],                      # Missing run ID
        ["predict"],                       # Missing path
        ["experiment", "--run-name", "x", "--epochs", "abc"],
        ["predict", "--path", "nonexistent.png"],
    ],
)
def test_invalid_cli_arguments(cli_setup, arguments):
    result = runner.invoke(main.app, arguments)

    assert result.exit_code != 0

    for mock in cli_setup.values():
        mock.assert_not_called()



## testing --help works ###

def test_cli_help():
    result = runner.invoke(main.app, ["--help"])

    assert result.exit_code == 0

    for command in [
        "experiment",
        "train",
        "evaluate",
        "register",
        "predict",
    ]:
        assert command in result.stdout
