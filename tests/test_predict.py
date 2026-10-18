import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from unittest.mock import MagicMock

from gan_busters import config
from gan_busters.modeling import predict


@pytest.mark.parametrize(
    "probability, threshold, expected_label, expected_confidence",
    [
        (0.8, 0.5, "FAKE", 0.8),
        (0.2, 0.5, "REAL", 0.8),
        (0.5, 0.5, "FAKE", 0.5),
        (0.6, 0.7, "REAL", 0.4),
    ],
)
def test_predict_images_threshold(
    probability, threshold, expected_label, expected_confidence
):
    model = MagicMock(
        return_value=tf.constant([[probability]], dtype=tf.float32)
    )

    images = [tf.zeros((32, 32, 3), dtype=tf.uint8)]

    results = predict.predict_images(model, images, threshold)

    assert results == [{
        "label": expected_label,
        "class_probability": expected_confidence,
    }]


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_predict_images_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        predict.predict_images(MagicMock(), [], threshold)


def test_predict_images_empty():
    model = MagicMock()

    assert predict.predict_images(model, []) == []
    model.assert_not_called()


def test_predict_images_different_shapes():
    images = [
        tf.zeros((32, 32, 3), dtype=tf.uint8),
        tf.zeros((64, 64, 3), dtype=tf.uint8),
    ]

    model = MagicMock(
        side_effect=[
            tf.constant([[0.8]], dtype=tf.float32),
            tf.constant([[0.2]], dtype=tf.float32),
        ]
    )

    results = predict.predict_images(model, images)

    assert [result["label"] for result in results] == ["FAKE", "REAL"]
    assert model.call_count == 2

    for call in model.call_args_list:
        assert call.args[0].shape[0] == 1


def test_predict_images_same_shapes():
    images = [
        tf.zeros((32, 32, 3), dtype=tf.uint8),
        tf.zeros((32, 32, 3), dtype=tf.uint8),
    ]

    model = MagicMock(
        return_value=tf.constant([[0.8], [0.2]], dtype=tf.float32)
    )

    results = predict.predict_images(model, images)

    assert [result["label"] for result in results] == ["FAKE", "REAL"]
    model.assert_called_once()
    assert model.call_args.args[0].shape == (2, 32, 32, 3)


@pytest.fixture
def predict_setup(monkeypatch, tmp_path):
    model = MagicMock(
        return_value=tf.constant([[0.8]], dtype=tf.float32)
    )

    mock_load_model = MagicMock(return_value=model)
    monkeypatch.setattr(
        predict.keras.models,
        "load_model",
        mock_load_model,
    )

    monkeypatch.setattr(
        config,
        "PREDICTIONS_DIR",
        tmp_path / "predictions",
    )

    return model, mock_load_model


def test_predict_from_path_single_image(sample_image, predict_setup):
    model, mock_load_model = predict_setup

    results = predict.predict_from_path(sample_image)

    assert len(results) == 1
    assert results[0]["image_path"] == str(sample_image)
    assert results[0]["label"] in ["REAL", "FAKE"]

    mock_load_model.assert_called_once_with(
        config.MODELS_DIR / config.DEFAULT_MODEL_NAME
    )

    output_path = config.PREDICTIONS_DIR / config.DEFAULT_PREDICTIONS_FILE
    assert output_path.exists()

    saved = pd.read_csv(output_path)
    assert len(saved) == 1


def test_predict_from_path_invalid_file(sample_non_image, predict_setup):
    with pytest.raises(ValueError, match="No valid images found"):
        predict.predict_from_path(sample_non_image)


def test_predict_from_path_invalid_path(tmp_path, predict_setup):
    missing_path = tmp_path / "missing.png"

    with pytest.raises(ValueError, match="Path must be an image file or directory"):
        predict.predict_from_path(missing_path)


def test_predict_from_path_directory(
    sample_image, sample_non_image, predict_setup, tmp_path
):
    import shutil

    directory = tmp_path / "images"
    directory.mkdir()

    shutil.copy(sample_image, directory / "valid.png")
    shutil.copy(sample_non_image, directory / "invalid.jpg")

    results = predict.predict_from_path(directory)

    assert len(results) == 1
    assert results[0]["image_path"] == str(directory / "valid.png")

    output_path = config.PREDICTIONS_DIR / config.DEFAULT_PREDICTIONS_FILE
    saved = pd.read_csv(output_path)

    assert len(saved) == 1
    assert saved.iloc[0]["image_path"] == str(directory / "valid.png")
