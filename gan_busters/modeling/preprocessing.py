"""
Image preprocessing utilities for GAN-Busters.

Provides dataset construction utilities and serializable preprocessing
components applied consistently during training and inference.
"""

from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from gan_busters.config import RAW_DATA_DIR


# -------------------------------------------------------------------------
# Model preprocessing
# -------------------------------------------------------------------------

@keras.utils.register_keras_serializable(package="gan_busters")
class ChannelStandardization(keras.layers.Layer):
    """
    Convert grayscale, RGB, or RGBA images to RGB.

    Rules:
        - 1 channel: replicate grayscale values across RGB channels.
        - 3 channels: preserve RGB values unchanged.
        - 4 channels: composite RGBA values onto a white background.

    Output pixel values remain in the [0, 255] range.
    """

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        channels = tf.shape(inputs)[-1]

        valid_channels = tf.reduce_any(
            tf.equal(channels, [1, 3, 4])
        )

        tf.debugging.assert_equal(
            valid_channels,
            True,
            message="Images must have 1, 3, or 4 channels.",
        )

        def grayscale_to_rgb():
            return tf.image.grayscale_to_rgb(inputs)

        def keep_rgb():
            return inputs

        def rgba_to_rgb():
            rgb = inputs[..., :3]
            alpha = inputs[..., 3:4] / 255.0
            white = tf.ones_like(rgb) * 255.0

            return alpha * rgb + (1.0 - alpha) * white

        return tf.case(
            [
                (tf.equal(channels, 1), grayscale_to_rgb),
                (tf.equal(channels, 4), rgba_to_rgb),
            ],
            default=keep_rgb,
            exclusive=True,
        )

    def compute_output_shape(self, input_shape):
        """Declare that the output always contains three channels."""
        return (*input_shape[:-1], 3)

    def get_config(self):
        """Return layer configuration for serialization."""
        return super().get_config()


# -------------------------------------------------------------------------
# Dataset construction
# -------------------------------------------------------------------------

def _load_image(
    path: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Read and decode one image without applying model preprocessing."""

    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=0,
        expand_animations=False,
    )

    image.set_shape([None, None, None])

    return image, label


def build_dataset(
    records: pd.DataFrame,
    batch_size: int,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """
    Build a TensorFlow dataset from validated image records.

    Images are decoded only. Channel standardization, resizing, cropping,
    and normalization are handled by the model itself.
    """

    image_paths = [
        str(RAW_DATA_DIR / Path(relative_path))
        for relative_path in records["relative_path"]
    ]

    labels = (
        records["label"]
        .map({"REAL": 0, "FAKE": 1})
        .astype("float32")
        .to_numpy()
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(records),
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        _load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset