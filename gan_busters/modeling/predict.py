from pathlib import Path
from typing import Annotated, Any

from loguru import logger
import pandas as pd
import tensorflow as tf
import typer

from gan_busters import config

app = typer.Typer(help="Inference module for GAN-Busters image classification.")


def predict_images(
    images: tf.Tensor | list[tf.Tensor],
    model_path: Path = config.MODELS_DIR / "gan_busters.keras",
) -> list[dict[str, Any]]:
    """Reusable inference function for in-memory decoded image tensors."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model artifact not found at: {model_path}")

    model = tf.keras.models.load_model(model_path)

    if isinstance(images, list):
        batch = tf.stack(images)
    elif len(images.shape) == 3:
        batch = tf.expand_dims(images, axis=0)
    else:
        batch = images

    probabilities = model.predict(batch, verbose=0).flatten()

    predictions = []
    for prob in probabilities:
        fake_prob = float(prob)
        label = "FAKE" if fake_prob >= 0.5 else "REAL"
        confidence = fake_prob if label == "FAKE" else (1.0 - fake_prob)
        predictions.append({
            "label": label,
            "confidence_score": round(confidence, 4),
            "fake_probability": round(fake_prob, 4),
        })

    return predictions


@app.command()
def main(
    image_path: Annotated[
        Path | None,
        typer.Option(
            "--image-path",
            "-i",
            help="Path to an image file or directory of images for inference.",
        ),
    ] = None,
    model_path: Annotated[
        Path,
        typer.Option(
            "--model-path",
            "-m",
            help="Path to the saved .keras model.",
        ),
    ] = config.MODELS_DIR / "gan_busters.keras",
    predictions_path: Annotated[
        Path,
        typer.Option(
            "--predictions-path",
            "-o",
            help="Destination path to export prediction results as CSV.",
        ),
    ] = config.PROCESSED_DATA_DIR / "test_predictions.csv",
):
    """CLI command to perform batch or single-image inference from disk."""
    if not image_path or not image_path.exists():
        logger.error(f"Provided path does not exist: {image_path}")
        raise typer.BadParameter("A valid --image-path must be specified.")

    if image_path.is_file():
        file_paths = [image_path]
    else:
        valid_extensions = {".jpg", ".jpeg", ".png"}
        file_paths = [p for p in image_path.rglob("*") if p.suffix.lower() in valid_extensions]

    if not file_paths:
        logger.warning(f"No valid image files found in: {image_path}")
        return

    logger.info(f"Decoding {len(file_paths)} image(s) in memory for inference...")

    decoded_tensors = []
    valid_paths = []

    for path in file_paths:
        try:
            img_bytes = tf.io.read_file(str(path))
            img_tensor = tf.io.decode_image(img_bytes, expand_animations=False)
            decoded_tensors.append(img_tensor)
            valid_paths.append(str(path))
        except (OSError, tf.errors.OpError) as e:
            logger.warning(f"Failed to decode image {path}: {e}")

    results = predict_images(decoded_tensors, model_path=model_path)

    records = []
    for path, res in zip(valid_paths, results):
        records.append({
            "image_path": path,
            "label": res["label"],
            "confidence_score": res["confidence_score"],
            "fake_probability": res["fake_probability"],
        })

    results_df = pd.DataFrame(records)

    if len(records) == 1:
        single = records[0]
        logger.success(
            f"Prediction: {single['label']} | "
            f"Confidence: {single['confidence_score']:.4f} | "
            f"P(FAKE): {single['fake_probability']:.4f}"
        )
    else:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(predictions_path, index=False)
        logger.success(f"Processed {len(records)} images. Saved predictions to: {predictions_path}")


if __name__ == "__main__":
    app()