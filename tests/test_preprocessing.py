import pytest
import numpy as np
import pandas as pd
import tensorflow as tf

from gan_busters.modeling import preprocessing
from gan_busters import config

# test when tensorflow knows the channels beforehand (static)
@pytest.mark.parametrize(
    "pixel, expected",
    [
        ([100], [100, 100, 100]),
        ([100, 150, 200], [100, 150, 200]),
        ([100, 150, 200, 255], [100, 150, 200]),
        ([100, 150, 200, 0], [255, 255, 255]),
    ],
)
def test_channel_standardization(pixel, expected):
    layer = preprocessing.ChannelStandardization()

    image = tf.constant([[pixel]], dtype=tf.uint8)
    result = layer(image).numpy()

    assert result.shape == (1, 1, 3) # check the output has 3 channels regardless of the original image format
    np.testing.assert_allclose(result[0, 0], expected)


# test when tensorflow doesnt know channels before applying the channel standardization layer
@pytest.mark.parametrize(
    "pixel, expected",
    [
        ([100], [100, 100, 100]),
        ([100, 150, 200], [100, 150, 200]),
        ([100, 150, 200, 255], [100, 150, 200]),
        ([100, 150, 200, 0], [255, 255, 255]),
    ],
)
def test_channel_standardization_dynamic(pixel, expected):
    layer = preprocessing.ChannelStandardization()

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[None, None, None, None],
                dtype=tf.uint8,
            )
        ]
    )
    def standardize(image):
        return layer(image)

    image = tf.constant([[[pixel]]], dtype=tf.uint8)
    result = standardize(image).numpy()

    assert result.shape == (1, 1, 1, 3)
    np.testing.assert_allclose(result[0, 0, 0], expected)


def test_load_image(sample_image):
    image = preprocessing.load_image(tf.constant(str(sample_image)))

    assert isinstance(image, tf.Tensor)
    assert image.shape == (32, 32, 3)
    assert image.dtype == tf.uint8


def test_build_dataset(sample_image):
    # create 2 records with diff labels using the same sample image
    records = pd.DataFrame({
        "relative_path": [sample_image.name, sample_image.name],
        "label": ["REAL", "FAKE"],
    })

    dataset = preprocessing.build_dataset(
        records,
        batch_size=2,
        image_dir=sample_image.parent,
    )

    images, labels = next(iter(dataset))

    assert images.shape == (2, 32, 32, 3)
    assert labels.shape == (2,)
    np.testing.assert_array_equal(
        labels.numpy(),
        [config.LABEL_MAPPING["REAL"], config.LABEL_MAPPING["FAKE"]],
    )


# testing that with the current implementation heterogenous images will raise an error at build_Dataset()
def test_build_dataset_incompatible_shapes(tmp_path):
    # Create two images with different dimensions
    shapes = [(32, 32, 3), (64, 64, 3)]
    paths = []

    for i, shape in enumerate(shapes):
        image = tf.zeros(shape, dtype=tf.uint8)
        path = tmp_path / f"image_{i}.png"

        tf.io.write_file(str(path), tf.io.encode_png(image))
        paths.append(path.name)

    records = pd.DataFrame({
        "relative_path": paths,
        "label": ["REAL", "FAKE"],
    })

    dataset = preprocessing.build_dataset(
        records,
        batch_size=2,
        image_dir=tmp_path,
    )

    # Batching should fail because image dimensions are incompatible
    with pytest.raises(tf.errors.InvalidArgumentError):
        next(iter(dataset))