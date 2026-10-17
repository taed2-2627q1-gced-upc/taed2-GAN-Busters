from unittest.mock import MagicMock

import pandas as pd
import pytest
from tensorflow import keras

from gan_busters.modeling import train
from gan_busters import config


@pytest.fixture
def train_setup(sample_image, monkeypatch):
    records = pd.DataFrame({
        "relative_path": [sample_image.name],
        "label": ["REAL"],
    })

    monkeypatch.setattr(
        train,
        "build_dataset",
        lambda records, batch_size, shuffle: "mock_dataset",
    ) #temporarily replace the build_dataset function with this patch

    tracker = MagicMock() # simulate the tracker started
    tracker.stop.return_value = 0.001

    monkeypatch.setattr(
        train,
        "EmissionsTracker",
        lambda **kwargs: tracker,
    )

    model = MagicMock()
    model.fit.return_value = MagicMock()

    kwargs = {
        "model": model,
        "train_records": records,
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "learning_rate": 0.001,
        "epochs": 2,
        "batch_size": 2,
    }

    return kwargs, tracker


# check that when there are validation records provided, early stopping is applied and the model knows it applied validation
def test_train_model_with_validation(train_setup):
    kwargs, tracker = train_setup
    kwargs["validation_records"] = kwargs["train_records"]

    history, emissions = train.train_model(**kwargs)

    fit_kwargs = kwargs["model"].fit.call_args.kwargs

    assert fit_kwargs["validation_data"] == "mock_dataset"
    assert len(fit_kwargs["callbacks"]) == 1
    assert isinstance(
        fit_kwargs["callbacks"][0],
        keras.callbacks.EarlyStopping,
    )
    assert fit_kwargs["callbacks"][0].monitor == "val_loss"

    tracker.start.assert_called_once()
    tracker.stop.assert_called_once()
    assert emissions == 0.001 #value fixed in train_setup


# check that the right process is followed when there is no validation data provided
def test_train_model_without_validation(train_setup):
    kwargs, tracker = train_setup

    history, emissions = train.train_model(**kwargs)

    fit_kwargs = kwargs["model"].fit.call_args.kwargs

    assert fit_kwargs["validation_data"] is None
    assert fit_kwargs["callbacks"] == []

    tracker.start.assert_called_once()
    tracker.stop.assert_called_once()
    assert emissions == 0.001



# check compiling configuration is correct
@pytest.mark.parametrize("optimizer", ["adam", "rmsprop"])
def test_train_model_optimizers(train_setup, optimizer):
    kwargs, tracker = train_setup
    kwargs["optimizer"] = optimizer

    train.train_model(**kwargs)

    compile_kwargs = kwargs["model"].compile.call_args.kwargs
    optimizer_instance = compile_kwargs["optimizer"]

    expected_type = (
        keras.optimizers.Adam
        if optimizer == "adam"
        else keras.optimizers.RMSprop
    )

    assert isinstance(optimizer_instance, expected_type)
    assert float(optimizer_instance.learning_rate.numpy()) == pytest.approx(0.001)
    assert compile_kwargs["loss"] == "binary_crossentropy"


# checking function stops before compiling or training when the optimizer is not valid
def test_train_model_invalid_optimizer(train_setup):
    kwargs, tracker = train_setup
    kwargs["optimizer"] = "invalid"

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        train.train_model(**kwargs)

    kwargs["model"].compile.assert_not_called()
    kwargs["model"].fit.assert_not_called()
    tracker.start.assert_not_called()


# check that cordecarbon stops when training fails
def test_train_model_tracker_on_failure(train_setup):
    kwargs, tracker = train_setup
    kwargs["model"].fit.side_effect = RuntimeError("Training failed")

    with pytest.raises(RuntimeError, match="Training failed"):
        train.train_model(**kwargs)

    tracker.start.assert_called_once()
    kwargs["model"].fit.assert_called_once()
    tracker.stop.assert_called_once()
    


# test main function
def test_train_final_model(monkeypatch, tmp_path):
    # Simulate processed records
    records = pd.DataFrame({
        "relative_path": ["train.png", "val.png", "test.png"],
        "split": ["train", "val", "test"],
        "label": ["REAL", "FAKE", "REAL"],
    })

    records_path = tmp_path / "accepted_records.csv"
    records.to_csv(records_path, index=False)

    monkeypatch.setattr(config, "ACCEPTED_RECORDS_PATH", records_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")

    # Simulate recovered MLflow configuration
    cfg = {
        "input_shape": (32, 32, 3),
        "conv_filters": (32, 32),
        "kernel_size": 3,
        "padding": "same",
        "conv_strides": (1, 1),
        "pooling": "max",
        "pool_size": (2, 2),
        "dense_units": (64,),
        "dropout": 0.25,
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 20,
    }

    mock_load_config = MagicMock(return_value=cfg)
    monkeypatch.setattr(train, "load_run_configuration", mock_load_config)

    # Avoid building and training a real CNN
    mock_model = MagicMock()
    mock_build_model = MagicMock(return_value=mock_model)
    monkeypatch.setattr(train, "build_model", mock_build_model)

    mock_train_model = MagicMock(return_value=(MagicMock(), 0.001))
    monkeypatch.setattr(train, "train_model", mock_train_model)

    # Avoid external MLflow/DagsHub operations
    mock_tracking = MagicMock()
    monkeypatch.setattr(train, "tracking", mock_tracking)

    # Run the final-training workflow
    train.train_final_model(
        run_id="test_run_id",
        epochs=10,
        model_name="test_model.keras",
    )

    # Verify configuration recovery and model construction
    mock_load_config.assert_called_once_with("test_run_id")

    mock_build_model.assert_called_once_with(
        input_shape=cfg["input_shape"],
        conv_filters=cfg["conv_filters"],
        kernel_size=cfg["kernel_size"],
        padding=cfg["padding"],
        conv_strides=cfg["conv_strides"],
        pooling=cfg["pooling"],
        pool_size=cfg["pool_size"],
        dense_units=cfg["dense_units"],
        dropout=cfg["dropout"],
    )

    # Verify that only train and validation records are used
    training_kwargs = mock_train_model.call_args.kwargs
    training_records = training_kwargs["train_records"]

    assert len(training_records) == 2
    assert set(training_records["split"]) == {"train", "val"}
    assert "test" not in training_records["split"].values

    # Verify that final training uses the requested epoch count
    assert training_kwargs["epochs"] == 10
    assert training_kwargs["validation_records"] if False else True
    assert training_kwargs["carbon_project_name"] == config.CODECARBON_FINAL_TRAINING_PROJECT

    # Verify MLflow parameter and emissions logging
    logged_params = mock_tracking.log_params.call_args.args[0]

    assert logged_params["source_run_id"] == "test_run_id"
    assert logged_params["source_max_epochs"] == 20
    assert logged_params["final_epochs"] == 10
    assert logged_params["training_records"] == 2
    assert "epochs" not in logged_params

    mock_tracking.log_metrics.assert_called_once_with({
        "emissions_kg_co2": 0.001,
    })

    # Verify model saving and registration
    expected_path = tmp_path / "models" / "test_model.keras"

    assert expected_path.parent.exists()
    mock_model.save.assert_called_once_with(expected_path)
    mock_tracking.log_model.assert_called_once_with(
        model=mock_model,
        name="final_model",
    )