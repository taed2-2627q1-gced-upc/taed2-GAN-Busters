
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from gan_busters import config
from gan_busters.modeling import evaluate


@pytest.fixture
def evaluation_setup(monkeypatch):
    records = pd.DataFrame({
        "relative_path": [
            "real_1.png", "real_2.png",
            "fake_1.png", "fake_2.png",
        ],
        "label": ["REAL", "REAL", "FAKE", "FAKE"],
    })

    dataset = MagicMock()
    monkeypatch.setattr(
        evaluate, "build_dataset",
        MagicMock(return_value=dataset),
    )

    model = MagicMock()
    model.predict.return_value = np.array([
        [0.1], [0.6], [0.8], [0.9],
    ])
    model.evaluate.return_value = 0.25 # value to be returned when loss is called

    return model, records, dataset


@pytest.mark.parametrize(
    "labels",
    [
        ["REAL", "REAL"],
        ["FAKE", "FAKE"],
        ["REAL", "FAKE", "OTHER"],
        ["REAL", "UNKNOWN"],
    ],
)
def test_evaluate_model_invalid_labels(evaluation_setup, labels):
    model, records, _ = evaluation_setup

    records = pd.DataFrame({"label": labels})

    with pytest.raises(ValueError, match="both REAL and FAKE"):
        evaluate.evaluate_model(model, records)


def test_evaluate_model_metrics(evaluation_setup):
    model, records, dataset = evaluation_setup

    metrics = evaluate.evaluate_model(model, records, threshold=0.5)

    model.predict.assert_called_once_with(dataset, verbose=0)
    model.evaluate.assert_called_once_with(dataset, verbose=0)

    assert metrics["loss"] == pytest.approx(0.25)
    assert metrics["real_precision"] == pytest.approx(1.0)
    assert metrics["real_recall"] == pytest.approx(0.5)
    assert metrics["fake_precision"] == pytest.approx(2 / 3)
    assert metrics["fake_recall"] == pytest.approx(1.0)

    # confirm f1 is the harmonic mean of precision and recall.
    for label in ["real", "fake"]:
        precision = metrics[f"{label}_precision"]
        recall = metrics[f"{label}_recall"]

        expected = 2 * precision * recall / (precision + recall)
        assert metrics[f"{label}_f1"] == pytest.approx(expected)

    # confirm macro F1 is the arithmetic mean of both class F1 scores.
    assert metrics["macro_f1"] == pytest.approx(
        (metrics["real_f1"] + metrics["fake_f1"]) / 2
    )

    assert metrics["roc_auc"] == pytest.approx(1.0) #all fake images have higher probability of being fake than real images


def test_evaluate_model_threshold(evaluation_setup):
    model, records, _ = evaluation_setup

    metrics_05 = evaluate.evaluate_model(model, records, threshold=0.5)
    metrics_07 = evaluate.evaluate_model(model, records, threshold=0.7)

    assert metrics_07["macro_f1"] > metrics_05["macro_f1"] #bc real2.png has a prob of 0.6 of being fake
    assert metrics_07["macro_f1"] == pytest.approx(1.0)

    # ROC-AUC uses probabilities, not thresholded predictions.
    assert metrics_07["roc_auc"] == metrics_05["roc_auc"]


def test_evaluate_model_batch_size(evaluation_setup, monkeypatch):
    model, records, _ = evaluation_setup

    mock_build_dataset = MagicMock(return_value="mock_dataset")
    monkeypatch.setattr(evaluate, "build_dataset", mock_build_dataset)

    evaluate.evaluate_model(model, records, batch_size=1)

    mock_build_dataset.assert_called_once_with(
        records=records,
        batch_size=1,
        shuffle=False,
    )

    
def test_evaluate_final_model(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / "final.keras"
    model_path.touch()

    monkeypatch.setattr(config, "MODELS_DIR", models_dir)

    records = pd.DataFrame({
        "relative_path": [
            "train.png", "val.png", "real.png", "fake.png"
        ],
        "split": ["train", "val", "test", "test"],
        "label": ["REAL", "FAKE", "REAL", "FAKE"],
    })

    records_path = tmp_path / "accepted.csv"
    records.to_csv(records_path, index=False)
    monkeypatch.setattr(config, "ACCEPTED_RECORDS_PATH", records_path)

    model = MagicMock()
    mock_load = MagicMock(return_value=model)
    monkeypatch.setattr(
        evaluate.keras.models, "load_model", mock_load
    )

    mock_evaluate = MagicMock(return_value={
        "loss": 0.2,
        "macro_f1": 0.9,
        "roc_auc": 0.95,
    })
    monkeypatch.setattr(evaluate, "evaluate_model", mock_evaluate)

    mock_tracking = MagicMock()
    monkeypatch.setattr(evaluate, "tracking", mock_tracking)

    metrics = evaluate.evaluate_final_model(
        model_name="final.keras",
        threshold=0.6,
        batch_size=16,
    )

    mock_load.assert_called_once_with(model_path)

    kwargs = mock_evaluate.call_args.kwargs
    test_records = kwargs["evaluation_records"]

    assert kwargs["model"] is model
    assert kwargs["threshold"] == 0.6
    assert kwargs["batch_size"] == 16
    assert len(test_records) == 2
    assert test_records["split"].tolist() == ["test", "test"]
    assert test_records["label"].tolist() == ["REAL", "FAKE"]

    assert metrics["test_loss"] == 0.2
    assert "loss" not in metrics

    mock_tracking.initialize_mlflow.assert_called_once()
    mock_tracking.set_experiment.assert_called_once_with(
        config.FINAL_EVALUATION_MLFLOW_EXPERIMENT
    )
    mock_tracking.start_run.assert_called_once_with(
        config.FINAL_EVALUATION_MLFLOW_RUN_NAME
    )
    mock_tracking.log_params.assert_called_once_with({
        "model_name": "final.keras",
        "threshold": 0.6,
        "batch_size": 16,
        "test_records": 2,
    })
    mock_tracking.log_metrics.assert_called_once_with(metrics)


def test_evaluate_final_model_missing_model(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="Saved model not found"):
        evaluate.evaluate_final_model(model_name="missing.keras")