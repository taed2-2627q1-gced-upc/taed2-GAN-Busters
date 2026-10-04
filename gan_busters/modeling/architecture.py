"""
CNN architecture construction for GAN-Busters.

This module builds the complete TensorFlow/Keras classification model,
including input preprocessing and the CNN architecture.

All model configuration is provided by the experiment runner. This module
does not define experiment defaults or training configuration.
"""

from tensorflow import keras
from tensorflow.keras import layers

from gan_busters.modeling.preprocessing import ChannelStandardization


def build_model(
    input_shape: tuple[int, int, int],
    conv_filters: tuple[int, ...],
    kernel_size: int,
    padding: str,
    conv_strides: tuple[int, int],
    pooling: str,
    pool_size: tuple[int, int],
    dense_units: tuple[int, ...],
    dropout: float,
) -> keras.Model:
    """Build and return the complete GAN-Busters CNN model."""

    # ---------------------------------------------------------
    # Input
    # ---------------------------------------------------------
    inputs = keras.Input(
        shape=(None, None, None),
        name="image",
    )

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------
    x = ChannelStandardization(
        name="channel_standardization",
    )(inputs)

    x = layers.Resizing(
        height=input_shape[0],
        width=input_shape[1],
        interpolation="bilinear",
        crop_to_aspect_ratio=True,
        name="resize",
    )(x)

    x = layers.Rescaling(
        scale=1.0 / 255.0,
        name="normalization",
    )(x)

    # ---------------------------------------------------------
    # Convolutional blocks
    # ---------------------------------------------------------
    for i, filters in enumerate(conv_filters, start=1):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=conv_strides,
            padding=padding,
            activation="relu",
            name=f"conv_{i}",
        )(x)

        if pooling == "max":
            x = layers.MaxPooling2D(
                pool_size=pool_size,
                name=f"max_pool_{i}",
            )(x)
        elif pooling == "avg":
            x = layers.AveragePooling2D(
                pool_size=pool_size,
                name=f"avg_pool_{i}",
            )(x)
        else:
            raise ValueError(
                f"Unsupported pooling type: {pooling}. "
                "Expected 'max' or 'avg'."
            )

    # ---------------------------------------------------------
    # Fully connected network
    # ---------------------------------------------------------
    x = layers.Flatten(name="flatten")(x)

    for i, units in enumerate(dense_units, start=1):
        x = layers.Dense(
            units=units,
            activation="relu",
            name=f"dense_{i}",
        )(x)

        if dropout > 0:
            x = layers.Dropout(
                rate=dropout,
                name=f"dropout_{i}",
            )(x)

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    outputs = layers.Dense(
        units=1,
        activation="sigmoid",
        name="fake_probability",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="gan_busters_cnn",
    )