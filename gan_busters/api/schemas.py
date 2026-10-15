from pydantic import BaseModel, Field
from typing import List, Literal

class ImagePrediction(BaseModel):
    filename: str = Field(..., description="Name of the uploaded image file.")
    label: Literal["REAL", "FAKE"] = Field(..., description="Predicted class for the image.")
    fake_probability: float = Field(..., ge=0.0, le=1.0, description="Probability score for the FAKE class.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the predicted class.")

class BatchPredictionResponse(BaseModel):
    predictions: List[ImagePrediction] = Field(..., description="List of results for each uploaded image.")

class HTTPErrorDetail(BaseModel):
    detail: str = Field(..., description="Explanation of the HTTP error.")