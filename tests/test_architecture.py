import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from gan_busters.modeling.architecture import build_model


@pytest.fixture
def model_config():
    return {
        "input_shape": (32, 32, 3),
        "conv_filters": (32, 32),
        "kernel_size": 3,
        "padding": "same",
        "conv_strides": (1, 1),
        "pooling": "avg",
        "pool_size": (2, 2),
        "dense_units": (64,),
        "dropout": 0.0,
    }


def test_build_model(model_config):
    model = build_model(**model_config)

    # Model construction
    assert isinstance(model, keras.Model)
    assert model.name == "gan_busters_cnn"

    # Preprocessing layers
    assert model.get_layer("channel_standardization") is not None
    assert model.get_layer("resize").height == 32
    assert model.get_layer("resize").width == 32
    assert model.get_layer("normalization").scale == 1.0 / 255.0

    # CNN architecture
    conv_layers = [
        layer for layer in model.layers
        if isinstance(layer, keras.layers.Conv2D)
    ]
    assert len(conv_layers) == 2
    assert all(layer.filters == 32 for layer in conv_layers)
    assert model.get_layer("dense_1").units == 64

    # Binary classification output
    output_layer = model.get_layer("fake_probability")
    assert output_layer.units == 1
    assert output_layer.activation.__name__ == "sigmoid"

    # Forward pass without compilation
    images = tf.zeros((2, 32, 32, 3), dtype=tf.uint8)
    predictions = model(images, training=False).numpy()

    assert predictions.shape == (2, 1)
    assert np.all((predictions >= 0) & (predictions <= 1))


@pytest.mark.parametrize("pooling", ["max", "avg"])
@pytest.mark.parametrize("dropout", [0.0, 0.25])
def test_build_model_configurations(model_config, pooling, dropout):
    model_config["pooling"] = pooling
    model_config["dropout"] = dropout

    model = build_model(**model_config)

    # Verify selected pooling type
    pooling_type = (
        keras.layers.MaxPooling2D
        if pooling == "max"
        else keras.layers.AveragePooling2D
    )

    pooling_layers = [
        layer for layer in model.layers
        if isinstance(layer, pooling_type)
    ]
    assert len(pooling_layers) == 2

    # Verify optional dropout
    dropout_layers = [
        layer for layer in model.layers
        if isinstance(layer, keras.layers.Dropout)
    ]

    assert len(dropout_layers) == (1 if dropout > 0 else 0)

    if dropout > 0:
        assert dropout_layers[0].rate == dropout


def test_build_model_invalid_pooling(model_config):
    model_config["pooling"] = "invalid"

    with pytest.raises(ValueError, match="Unsupported pooling type"):
        build_model(**model_config)


@pytest.mark.parametrize("shape", [
    (32, 32, 3),
    (64, 64, 3),
    (48, 64, 3),
])
def test_build_model_different_image_shapes(model_config, shape):
    model = build_model(**model_config)

    images = tf.zeros((1, *shape), dtype=tf.uint8)
    predictions = model(images, training=False).numpy()

    assert predictions.shape == (1, 1)
    assert np.all((predictions >= 0) & (predictions <= 1))