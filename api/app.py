from contextlib import asynccontextmanager
from typing import List, Literal
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
import tensorflow as tf
from tensorflow import keras

from gan_busters.config import MODELS_DIR, DEFAULT_THRESHOLD
from gan_busters.modeling.predict import predict_images

model: keras.Model = None

SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the previosly trained Keras model during the API startup"""
    global model
    model_path = MODELS_DIR / "gan_busters.keras"
    if not model_path.exists():
        raise RuntimeError(f"Model file couldn't be found at {model_path}")

    model = keras.models.load_model(model_path)
    yield

app = FastAPI(
    title="GAN-Busters AI Image Classification API",
    description=(
        "REST API for detecting AI-generated vs authentic images. "
        "Applies dinamic image preprocessing inside of the neural model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

class ImagePrediction(BaseModel):
    filename: str = Field(..., description="Name of the uploaded image file.")
    label: Literal["REAL", "FAKE"] = Field(..., description="Predicted class for the image.")
    fake_probability: float = Field(..., ge=0.0, le=1.0, description="Probability score for the FAKE class.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the predicted class.")

class BatchPredictionResponse(BaseModel):
    predictions: List[ImagePrediction] = Field(..., description="List of results for each uploaded image.")

class HTTPErrorDetail(BaseModel):
    detail: str = Field(..., description="Explanation of the HTTP error.")

@app.post(
    "/predict",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify multiple images as REAL or FAKE.",
    description="Accepts multiple images, validates format, preprocesses images and returns predictions.",
    responses={
        400: {"model": HTTPErrorDetail, "description": "Empty payload or corrupt image data."},
        415: {"model": HTTPErrorDetail, "description": "Unsupported media format, only " + repr(ALLOWED_EXTENSIONS) + " are allowed."},
        500: {"model": HTTPErrorDetail, "description": "Empty payload or corrupt image data."},
    },
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

    decoded_tensors: List[tf.Tensor] = []
    file_metadata: List[str] = []

    #Analyze each of the images
    for file in files:
        #Check the extension of the file, if there is one non supported file raise code 415
        ext = f".{file.filename.split('.')[-1].lower()}" if "." in file.filename else ""
        if file.content_type not in SUPPORTED_CONTENT_TYPES and ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported media file for file '{file.filename}'. Only {repr(ALLOWED_EXTENSIONS)} formats are allowed."
            )

        content = await file.read()

        if len(content) == 0:
            #If image has no content raise code 400
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{file.filename}' is empty."
            )

        #Decode image and check for corruption
        try:
            image_tensor = tf.io.decode_image(
                content,
                channels=0,
                expand_animations=False,
            )
            image_tensor.set_shape([None, None, None])

            decoded_tensors.append(image_tensor)
            file_metadata.append(file.filename)
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unreadable image file '{file.filename}': {str(err)}",
            )

    raw_results = predict_images(model=model, images=decoded_tensors, threshold=DEFAULT_THRESHOLD)

    predictions = [
        ImagePrediction(
            filename=filename,
            label=res["label"],
            fake_probability=res["fake_probability"],
            confidence_score=res["confidence_score"],
        )
        for filename, res in zip(file_metadata, raw_results)
    ]

    return BatchPredictionResponse(predictions=predictions)