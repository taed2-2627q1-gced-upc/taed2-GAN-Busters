"""
Inference utilities for GAN-Busters.

This module provides reusable prediction functionality for decoded images
held in memory.

The prediction logic is independent of the input source. Images may originate
from a CLI workflow, REST API upload, or another application layer, but must
be decoded before being passed to this module.

No input images are read from or written to persistent storage here.
"""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import tensorflow as tf
from loguru import logger
from tensorflow import keras

from gan_busters import config
from gan_busters.modeling.preprocessing import load_image


def predict_images(
    model: keras.Model,
    images: Sequence[tf.Tensor],
    threshold: float = config.DEFAULT_THRESHOLD,
) -> list[dict]:
    """
    Run inference on one or more decoded images.

    Args:
        model:
            Trained GAN-Busters Keras model.

        images:
            Sequence of decoded image tensors. Images may have different
            spatial dimensions and may contain 1, 3, or 4 channels.

        threshold:
            Probability threshold used to classify an image as FAKE.

    Returns:
        A list containing the predicted label and class probability
        for each image.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    if len(images) == 0:
        return []

    same_shape = all(
        image.shape == images[0].shape
        for image in images
    )

    if same_shape:
        probabilities = model(
            tf.stack(images),
            training=False,
        )
    else:
        probabilities = tf.concat(
            [
                model(
                    tf.expand_dims(image, axis=0),
                    training=False,
                )
                for image in images
            ],
            axis=0,
        )

    fake_probabilities = tf.reshape(
        probabilities,
        [-1],
    ).numpy()

    predictions = []

    for fake_probability in fake_probabilities:
        fake_probability = float(fake_probability)

        label = (
            "FAKE"
            if fake_probability >= threshold
            else "REAL"
        )

        class_probability = (
            fake_probability
            if label == "FAKE"
            else 1.0 - fake_probability
        )

        predictions.append(
            {
                "label": label,
                "class_probability": round(class_probability, 4)
            }
        )

    return predictions


def predict_from_path(
    path: Path,
    model_name: str = config.DEFAULT_MODEL_NAME,
    threshold: float = config.DEFAULT_THRESHOLD,
    output_file: str = config.DEFAULT_PREDICTIONS_FILE,
) -> list[dict]:
    """Run predictions on one image or all valid images in a directory."""

    model = keras.models.load_model(
        config.MODELS_DIR / model_name
    )

    if path.is_file():
        image_paths = [path]

    elif path.is_dir():
        image_paths = [
            image_path
            for image_path in path.iterdir()
            if image_path.is_file()
        ]

    else:
        raise ValueError(
            f"Path must be an image file or directory: {path}"
        )

    images = []
    valid_paths = []

    for image_path in image_paths:
        try:
            image = load_image(str(image_path))
            images.append(image)
            valid_paths.append(image_path)

        except tf.errors.InvalidArgumentError:
            logger.warning(
                f"Skipping invalid image file: {image_path}"
            )

    if not images:
        raise ValueError(
            f"No valid images found at: {path}"
        )

    predictions = predict_images(
        model=model,
        images=images,
        threshold=threshold,
    )

    results = [
        {
            "image_path": str(image_path),
            **prediction,
        }
        for image_path, prediction in zip(valid_paths, predictions)
    ]

    output_path = config.PREDICTIONS_DIR / output_file
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(results).to_csv(
        output_path,
        index=False,
    )

    return results