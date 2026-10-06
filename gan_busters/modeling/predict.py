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
import tensorflow as tf
from tensorflow import keras


def predict_images(
    model: keras.Model,
    images: Sequence[tf.Tensor],
    threshold: float = 0.5,
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
        A list containing the predicted label and P(FAKE) for each image.

    Notes:
        Channel standardization, cropping, resizing, and normalization are
        handled by the preprocessing layers embedded in the Keras model.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    if len(images) == 0:
        return []

    predictions = []

    for image in images:
        # Add the batch dimension expected by the Keras model.
        image_batch = tf.expand_dims(image, axis=0)

        probability = model(
            image_batch,
            training=False,
        )

        fake_probability = float(
            tf.reshape(probability, [-1])[0].numpy()
        )

        label = "FAKE" if fake_probability >= threshold else "REAL"
        confidence = fake_probability if label == "FAKE" else (1.0 - fake_probability)

        predictions.append(
            {
                "label": label,
                "confidence_score": round(confidence, 4),
                "fake_probability": round(fake_probability, 4),
            }
        )

    return predictions