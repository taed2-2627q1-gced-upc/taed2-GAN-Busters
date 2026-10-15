from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from loguru import logger

import tensorflow as tf
from tensorflow import keras
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from gan_busters.modeling.predict import predict_images

from gan_busters.config import (
    MODELS_DIR, 
    DEFAULT_THRESHOLD,
    MAX_FILES_PER_BATCH,
    MAX_BATCH_SIZE,
    MAX_FILE_SIZE,
    MAX_IMAGE_DIM
    )

from gan_busters.api.schemas import (
    ImagePrediction, 
    BatchPredictionResponse, 
    HTTPErrorDetail
)

model: keras.Model = None

SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FORMAT_TO_MIME = {"JPEG": "image/jpeg", "PNG": "image/png"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the previosly trained Keras model during the API startup"""
    global model
    model_path = MODELS_DIR / "gan_busters.keras"
    if not model_path.is_file():
        logger.error(f"Model file couldn't be found at {model_path}")
        raise RuntimeError(f"Model file is missing")

    try:
        model = keras.models.load_model(model_path)
        logger.info("Loaded GAN-Busters model from {}", model_path)
    except Exception:
        logger.exception('Failed to load GAN-Busters model')
        raise

    try:
        yield
    finally:
        model = None

app = FastAPI(
    title="GAN-Busters AI Image Classification API",
    description=(
        "REST API for detecting AI-generated vs authentic images. "
        "Applies dinamic image preprocessing inside of the neural model."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

@app.get("/", tags=["General"])
def root():
    return {"message": "Welcome to the gan-busters app"}

@app.get("/health/live", tags=["Health"])
def liveness():
    return {"status": "alive"}

@app.get(
        "/health/ready",
        tags=["Health"],
        responses={503: {"model": HTTPErrorDetail, "description": "Model is not ready."}}
        )
def readiness():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The model is not ready"
        )
    return {"status": "ready", "model_loaded": True}

async def _read_limited_file(file: UploadFile, remaining_batch_bytes: int) -> bytes:
    allowed_read = min(MAX_FILE_SIZE, remaining_batch_bytes)
    content = await file.read(allowed_read + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File '{file.filename}' exceeds the 10 MiB per-file limit.",
        )

    if len(content) > remaining_batch_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="The combined image content exceeds the 25 MiB batch limit.",
        )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file '{file.filename}' is empty.",
        )

    return content

def _validate_and_decode_image(filename: str, content_type: str, content: bytes) -> tf.Tensor:
    suffix = Path(filename).suffix.lower()
    declared_type = (content_type or "").split(";", 1)[0].strip().lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file extension for '{filename}'. Only JPEG and PNG are allowed.",
        )

    if declared_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type for '{filename}'. Use image/jpeg or image/png.",
        )

    try:
        # PIL inspects the real file format.
        with Image.open(BytesIO(content)) as image:
            actual_format = image.format
            width, height = image.size
            image.verify()
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError,
        SyntaxError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{filename}' is not a valid readable image.",
        ) from exc

    if actual_format not in FORMAT_TO_MIME:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image format in '{filename}'. Only JPEG and PNG are allowed.",
        )

    expected_mime = FORMAT_TO_MIME[actual_format]
    expected_extensions = {".jpg", ".jpeg"} if actual_format == "JPEG" else {".png"}

    if declared_type != expected_mime or suffix not in expected_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"The file extension, content type, and actual image format do not match "
                f"for '{filename}'."
            ),
        )

    if width <= 0 or height <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image '{filename}' has invalid dimensions.",
        )

    if (
        width > MAX_IMAGE_DIM
        or height > MAX_IMAGE_DIM
    ):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Image '{filename}' exceeds the dimension limit "
                f"({MAX_IMAGE_DIM}x{MAX_IMAGE_DIM})."
            ),
        )

    try:
        # TensorFlow performs the decode used by the model after PIL validation.
        tensor = tf.io.decode_image(
            content,
            channels=0,
            expand_animations=False,
        )
        tensor.set_shape([None, None, None])
        return tensor
    except (tf.errors.InvalidArgumentError, ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image '{filename}' could not be decoded.",
        ) from exc

@app.post(
    "/predict",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify multiple images as REAL or FAKE.",
    description="Accepts up to 10 JPEG/PNG images, validates format, preprocesses images and returns predictions.",
    responses={
        400: {"model": HTTPErrorDetail, "description": "Empty payload or corrupt image data."},
        413: {"model": HTTPErrorDetail, "description": "Uploda, batch or image dimensions exceed limits."},
        415: {"model": HTTPErrorDetail, "description": "Unsupported media format, only " + repr(ALLOWED_EXTENSIONS) + " are allowed."},
        500: {"model": HTTPErrorDetail, "description": "Empty payload or corrupt image data."},
        503: {"model": HTTPErrorDetail, "description": "Model is not ready."}
    },
    tags=["Predictions"]
)
async def predict_batch(
    files: List[UploadFile] = File(..., description="List of image files to classify.")
):
    #Raise code 400 if there are no images uploaded
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload must contain at least 1 image file.",
        )

    #Raise code 413 if limits exceeded
    if len(files) > MAX_FILES_PER_BATCH:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"A batch may contain at most {MAX_FILES_PER_BATCH} images."
        )

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The model is not ready."
        )

    decoded_tensors: List[tf.Tensor] = []
    filenames: List[str] = []
    total_bytes = 0

    #Analyze each of the images
    try:
        for file in files:
            filename = file.filename or "unnamed"
            remaining = MAX_BATCH_SIZE - total_bytes
            content = await _read_limited_file(file, remaining)
            total_bytes += len(content)

            image_tensor = _validate_and_decode_image(
                filename=filename,
                content_type=file.content_type or "",
                content=content
            )
            decoded_tensors.append(image_tensor)
            filenames.append(filename)

        try:
            raw_results = await run_in_threadpool(
                predict_images,
                model=model,
                images=decoded_tensors,
                threshold=DEFAULT_THRESHOLD,
            )
        except Exception:
            logger.exception("Model inference failed for a batch of {} images", len(files))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Image inference failed. Try again later."
            ) from None

        if len(raw_results) != len(filenames):
            logger.error(
                "Inference returned {} results for {} input images.",
                len(raw_results),
                len(filenames)
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server's number of predictions dont match files."
            )

        try:
            predictions = [
                ImagePrediction(
                    filename=filename,
                    label=result["label"],
                    fake_probability=result["fake_probability"],
                    confidence_score=result["confidence_score"]
                )
                for filename, result in zip(filenames, raw_results)
            ]
        except (KeyError, TypeError, ValueError) as exc:
            logger.exception("Inference results don't match the expected response schema")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The model returned an invalid prediction result."
            ) from exc

        return BatchPredictionResponse(predictions=predictions)

    finally:
        for file in files:
            await file.close()