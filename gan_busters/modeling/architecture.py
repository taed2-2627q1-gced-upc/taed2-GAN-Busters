import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore[import-untyped]

from gan_busters import config
from gan_busters.modeling.preprocessing import ChannelStandardization


def build_model(
    conv_filters=config.DEFAULT_CONV_FILTERS,
    kernel_size=config.DEFAULT_KERNEL_SIZE,
    padding=config.DEFAULT_PADDING,
    pooling=config.DEFAULT_POOLING,
    pool_size=config.DEFAULT_POOL_SIZE,
    dense_units=config.DEFAULT_DENSE_UNITS,
    dropout=config.DEFAULT_DROPOUT,
    learning_rate=config.DEFAULT_LEARNING_RATE
) -> tf.keras.Model:
    """
    Constructs a fresh Keras Model packaging the preprocessing layers
    together with the CNN architecture.
    """
    pool_layer = layers.MaxPooling2D if pooling == "max" else layers.AveragePooling2D

    model = models.Sequential(name="gan_busters_cnn")
    model.add(layers.Input(shape=(None, None, None), name="raw_image_input"))

    # Embedded Preprocessing Layers
    model.add(ChannelStandardization(name="channel_standardization"))
    model.add(layers.Resizing(height=32, width=32, interpolation="bilinear", crop_to_aspect_ratio=True, name="crop_and_resize"))
    
    if config.NORMALIZE_PIXELS:
        model.add(layers.Rescaling(1.0 / 255.0, name="pixel_normalization"))

    # Convolutional Layers
    for i, filters in enumerate(conv_filters):
        model.add(layers.Conv2D(
            filters=filters,
            kernel_size=(kernel_size, kernel_size),
            padding=padding,
            activation="relu",
            name=f"conv2d_{i+1}"
        ))

    model.add(pool_layer(pool_size=pool_size, name="pooling"))
    model.add(layers.Flatten(name="flatten"))

    # Dense Layers
    for j, units in enumerate(dense_units):
        model.add(layers.Dense(units, activation="relu", name=f"dense_{j+1}"))

    if dropout > 0.0:
        model.add(layers.Dropout(dropout, name="dropout"))

    # Binary Output
    model.add(layers.Dense(1, activation="sigmoid", name="output_sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=config.DEFAULT_LOSS,
        metrics=["accuracy"]
    )
    return model