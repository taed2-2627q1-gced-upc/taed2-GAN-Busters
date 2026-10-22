
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pandas as pd
import pytest

from gan_busters import config
from gan_busters.modeling import experiment


@pytest.fixture
def experiment_setup(monkeypatch, tmp_path):
    records = pd.DataFrame({
        "relative_path": [
            "train_real.png",
            "train_fake.png",
            "val_real.png",
            "val_fake.png",
            "test_real.png",
            "test_fake.png",
        ],
        "split": ["train", "train", "val", "val", "test", "test"],
        "label": ["REAL", "FAKE", "REAL", "FAKE", "REAL", "FAKE"],
    })

    records_path = tmp_path / "accepted.csv"
    records.to_csv(records_path, index=False)
    monkeypatch.setattr(config, "ACCEPTED_RECORDS_PATH", records_path)

    model = MagicMock()
    mock_build_model = MagicMock(return_value=model)
    monkeypatch.setattr(experiment, "build_model", mock_build_model)

    history = SimpleNamespace(history={
        "loss": [0.6, 0.4, 0.3],
        "val_loss": [0.5, 0.2, 0.3],
    })

    mock_train_model = MagicMock(return_value=(history, 0.001))
    monkeypatch.setattr(experiment, "train_model", mock_train_model)

    mock_evaluate_model = MagicMock(return_value={
        "loss": 0.25,
        "macro_f1": 0.9,
        "roc_auc": 0.95,
    })
    monkeypatch.setattr(experiment, "evaluate_model", mock_evaluate_model)

    mock_tracking = MagicMock()
    monkeypatch.setattr(experiment, "tracking", mock_tracking)

    mock_seed = MagicMock()
    monkeypatch.setattr(
        experiment.tf.keras.utils, "set_random_seed", mock_seed
    )

    monkeypatch.setattr(
        experiment.tf.config,
        "list_physical_devices",
        lambda device_type: [],
    )

    return {
        "model": model,
        "build_model": mock_build_model,
        "train_model": mock_train_model,
        "evaluate_model": mock_evaluate_model,
        "tracking": mock_tracking,
        "seed": mock_seed,
    }



def test_run_experiment(experiment_setup):
    mocks = experiment_setup

    experiment.run_experiment(
        run_name="test_run",
        experiment_name="test_experiment",
        kernel_size=5,
        padding="same",
        pooling="avg",
        learning_rate=0.0001,
        batch_size=64,
        dropout=0.25,
        epochs=10,
    )

    # Reproducibility
    mocks["seed"].assert_called_once_with(config.RANDOM_SEED)

    # Model architecture
    mocks["build_model"].assert_called_once_with(
        input_shape=config.INPUT_SHAPE,
        conv_filters=config.DEFAULT_CONV_FILTERS,
        kernel_size=5,
        padding="same",
        conv_strides=config.DEFAULT_CONV_STRIDES,
        pooling="avg",
        pool_size=config.DEFAULT_POOL_SIZE,
        dense_units=config.DEFAULT_DENSE_UNITS,
        dropout=0.25,
    )

    # Training configuration and data split
    train_kwargs = mocks["train_model"].call_args.kwargs

    assert train_kwargs["model"] is mocks["model"]
    assert train_kwargs["optimizer"] == config.DEFAULT_OPTIMIZER
    assert train_kwargs["loss"] == config.DEFAULT_LOSS
    assert train_kwargs["learning_rate"] == 0.0001
    assert train_kwargs["batch_size"] == 64
    assert train_kwargs["epochs"] == 10

    train_records = train_kwargs["train_records"]
    validation_records = train_kwargs["validation_records"]

    assert train_records["split"].tolist() == ["train", "train"]
    assert train_records["label"].tolist() == ["REAL", "FAKE"]

    assert validation_records["split"].tolist() == ["val", "val"]
    assert validation_records["label"].tolist() == ["REAL", "FAKE"]

    # Evaluation must use the same validation records
    eval_kwargs = mocks["evaluate_model"].call_args.kwargs

    assert eval_kwargs["model"] is mocks["model"]
    pd.testing.assert_frame_equal(
        eval_kwargs["evaluation_records"],
        validation_records,
    )

    # MLflow tracking
    tracking = mocks["tracking"]

    tracking.initialize_mlflow.assert_called_once()
    tracking.set_experiment.assert_called_once_with("test_experiment")
    tracking.start_run.assert_called_once_with("test_run")

    params = tracking.log_params.call_args.args[0]

    assert params["kernel_size"] == 5
    assert params["padding"] == "same"
    assert params["pooling"] == "avg"
    assert params["learning_rate"] == 0.0001
    assert params["batch_size"] == 64
    assert params["dropout"] == 0.25
    assert params["epochs"] == 10

    tracking.log_history.assert_called_once()

    assert tracking.log_metrics.call_args_list == [
        call({
            "emissions_kg_co2": 0.001,
            "best_epoch": 2,
        }),
        call({
            "validation_loss": 0.25,
            "macro_f1": 0.9,
            "roc_auc": 0.95,
        }),
    ]



@pytest.mark.parametrize(
    "val_losses, expected_epoch",
    [
        ([0.2, 0.3, 0.4], 1),
        ([0.5, 0.2, 0.3], 2),
        ([0.5, 0.4, 0.2], 3),
    ],
)
def test_run_experiment_best_epoch(
    experiment_setup, val_losses, expected_epoch
):
    mocks = experiment_setup

    history = SimpleNamespace(history={
        "val_loss": val_losses,
    })

    mocks["train_model"].return_value = (history, 0.001)

    experiment.run_experiment(run_name="test_run")

    mocks["tracking"].log_metrics.assert_any_call({
        "emissions_kg_co2": 0.001,
        "best_epoch": expected_epoch,
    })

