import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers  # type: ignore[import-untyped]

from gan_busters import config


@tf.keras.utils.register_keras_serializable(package="gan_busters")
class ChannelStandardization(layers.Layer):
    """
    Standardizes input channels to 3-channel RGB:
    - 1 channel (Grayscale) -> Converts to RGB
    - 3 channels (RGB) -> Keeps unchanged
    - 4 channels (RGBA) -> Alpha blends with a white background into RGB
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        channels = tf.shape(x)[-1]

        def handle_grayscale():
            return tf.image.grayscale_to_rgb(x)

        def handle_rgb():
            return x

        def handle_rgba():
            rgb = x[..., :3]
            alpha = x[..., 3:]
            max_alpha = tf.reduce_max(alpha)
            # Normalize alpha to [0, 1] if needed
            alpha_norm = tf.cond(max_alpha > 1.0, lambda: alpha / 255.0, lambda: alpha)
            white = tf.cond(tf.reduce_max(rgb) > 1.0, lambda: 255.0, lambda: 1.0)
            return rgb * alpha_norm + white * (1.0 - alpha_norm)

        return tf.cond(
            tf.equal(channels, 1),
            handle_grayscale,
            lambda: tf.cond(tf.equal(channels, 4), handle_rgba, handle_rgb)
        )


def _load_and_decode_image(rel_path, label):
    abs_path = config.PROJ_ROOT / rel_path
    img_bytes = tf.io.read_file(str(abs_path))
    img = tf.io.decode_image(img_bytes, expand_animations=False)
    return img, tf.cast(label, tf.float32)


def build_dataset(records_df: pd.DataFrame, batch_size: int = config.DEFAULT_BATCH_SIZE, shuffle: bool = True):
    """
    Builds an optimized tf.data.Dataset from accepted records DataFrame.
    Cropping, resizing, and normalization are deferred to the Keras model.
    """
    paths = records_df["relative_path"].values
    labels = records_df["label"].values  # 0 for REAL, 1 for FAKE

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(records_df), 10000), seed=config.RANDOM_SEED)

    dataset = dataset.map(_load_and_decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset