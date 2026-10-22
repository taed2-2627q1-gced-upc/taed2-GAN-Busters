"""Define Pydantic schemas for API prediction responses and errors."""

from typing import Literal

from pydantic import BaseModel, Field


class ImagePrediction(BaseModel):
    """Represent the classification result for a single image."""

    filename: str = Field(..., description="Name of the uploaded image file.")
    label: Literal["REAL", "FAKE"] = Field(..., description="Predicted class for the image.")
    fake_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability score for the FAKE class."
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the predicted class."
    )


class BatchPredictionResponse(BaseModel):
    """Represent the collection of predictions returned for an image batch."""

    predictions: list[ImagePrediction] = Field(
        ..., description="List of results for each uploaded image."
    )


class HTTPErrorDetail(BaseModel):
    """Represent an HTTP error response with a descriptive message."""

    detail: str = Field(..., description="Explanation of the HTTP error.")
