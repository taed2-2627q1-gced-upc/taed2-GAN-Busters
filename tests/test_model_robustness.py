
"""
Model robustness tests for GAN-Busters (NFR17).

Evaluate prediction invariance on a reproducible CIFAKE test subset,
stratified by label and semantic subclass, under three label-preserving
transformations: mild Gaussian noise, centered cropping with bilinear
resizing, and horizontal flipping.

Each transformation must preserve at least 95% of the original
model predictions to satisfy NFR17.
"""

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from tensorflow import keras
import mlflow

from gan_busters import config
from gan_busters.modeling.predict import predict_images
from gan_busters.modeling import tracking

# -------------------------------------------------------------------------
# Image transformations
# -------------------------------------------------------------------------

def apply_transformation(images, transformation):
    """Apply a label-preserving transformation to a batch of images."""

    if transformation == "gaussian_noise":
        rng = np.random.default_rng(config.RANDOM_SEED)

        noise = rng.normal(
            loc=0,
            scale=config.ROBUSTNESS_NOISE_STD * 255,
            size=images.shape,
        )

        return tf.convert_to_tensor(
            np.clip(images.numpy() + noise, 0, 255),
            dtype=tf.float32,
        )

    if transformation == "center_crop":
        cropped = images[:, 1:-1, 1:-1, :]

        return tf.image.resize(
            cropped,
            size=(32, 32),
            method="bilinear",
        )

    if transformation == "horizontal_flip":
        return tf.image.flip_left_right(images)

    raise ValueError(f"Unknown transformation: {transformation}")


# -------------------------------------------------------------------------
# Reproducible robustness subset and trained model
# -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def robustness_setup():
    records = pd.read_csv(config.ACCEPTED_RECORDS_PATH)
    test_records = records[records["split"] == "test"]

    # Select an equal number of images from each label-subclass stratum.
    groups = test_records.groupby(["label", "subclass"])

    assert config.ROBUSTNESS_SUBSET_SIZE % groups.ngroups == 0

    samples_per_stratum = (
        config.ROBUSTNESS_SUBSET_SIZE // groups.ngroups
    )

    selected = groups.sample(
        n=samples_per_stratum,
        random_state=config.RANDOM_SEED,
    )

    assert len(selected) == config.ROBUSTNESS_SUBSET_SIZE

    # Load original images
    images = tf.stack([
        tf.convert_to_tensor(
            np.asarray(
                keras.utils.load_img(
                    config.RAW_DATA_DIR / relative_path,
                    color_mode="rgb",
                ),
                dtype=np.float32,
            )
        )
        for relative_path in selected["relative_path"]
    ])

    # Load the registered final model from MLflow
    tracking.initialize_mlflow()

    model = mlflow.keras.load_model(
        model_uri=(
            f"models:/{config.REGISTERED_MODEL_NAME}/"
            f"{config.ROBUSTNESS_MODEL_VERSION}"
        )
    )

    return model, images, selected.reset_index(drop=True)


# -------------------------------------------------------------------------
# Prediction invariance (NFR17)
# -------------------------------------------------------------------------

@pytest.mark.robustness
@pytest.mark.parametrize(
    "transformation",
    [
        "gaussian_noise",
        "center_crop",
        "horizontal_flip",
    ],
)
def test_prediction_invariance(robustness_setup, transformation):
    model, images, selected = robustness_setup

    original_predictions = predict_images(
        model=model,
        images=list(tf.unstack(images)),
        threshold=config.DEFAULT_THRESHOLD,
    )

    transformed_images = apply_transformation(
        images,
        transformation,
    )

    transformed_predictions = predict_images(
        model=model,
        images=list(tf.unstack(transformed_images)),
        threshold=config.DEFAULT_THRESHOLD,
    )

    original_labels = np.array([
        prediction["label"]
        for prediction in original_predictions
    ])

    transformed_labels = np.array([
        prediction["label"]
        for prediction in transformed_predictions
    ])

    # Identify images whose predictions changed
    changed = original_labels != transformed_labels
    invariance = np.mean(~changed)

    # Attach prediction changes to the selected image metadata
    results = selected[["label", "subclass"]].copy()
    results["changed"] = changed

    # Summarize changes by target label
    label_summary = (
        results.groupby("label")["changed"]
        .agg(changed="sum", total="count", change_rate="mean")
    )

    # Summarize changes by target label and semantic subclass
    subclass_summary = (
        results.groupby(["label", "subclass"])["changed"]
        .agg(changed="sum", total="count", change_rate="mean")
    )

    # Display results
    print(f"\n{transformation}: {invariance:.2%} invariance")

    print("\nPrediction changes by target label:")
    print(label_summary.to_string(float_format=lambda x: f"{x:.2%}"))

    print("\nPrediction changes by target label and subclass:")
    print(subclass_summary.to_string(float_format=lambda x: f"{x:.2%}"))

    assert invariance >= config.ROBUSTNESS_MIN_INVARIANCE, (
        f"{transformation}: {invariance:.2%} invariance, "
        f"below the required "
        f"{config.ROBUSTNESS_MIN_INVARIANCE:.0%}"
    )
