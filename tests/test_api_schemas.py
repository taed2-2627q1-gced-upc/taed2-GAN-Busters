import pytest
from pydantic import ValidationError
from gan_busters.api.schemas import (
    ImagePrediction,
    BatchPredictionResponse,
    HTTPErrorDetail
)

def test_image_prediction_valid():
    prediction = ImagePrediction(
        filename="test.jpg",
        label="REAL",
        fake_probability=0.15,
        confidence_score=0.85,
    )
    assert prediction.filename == "test.jpg"
    assert prediction.label == "REAL"
    assert prediction.fake_probability == 0.15
    assert prediction.confidence_score == 0.85

@pytest.mark.parametrize("label", ["REAL", "FAKE"])
def test_image_prediction_labels(label):
    prediction = ImagePrediction(
        filename="img.jpg",
        label=label,
        fake_probability=0.5,
        confidence_score=0.5
    )
    assert prediction.label == label

def test_image_prediction_invalid_label():
    with pytest.raises(ValidationError):
        ImagePrediction(
            filename="img.png",
            label="UNKNOWN",
            fake_probability=0.5,
            confidence_score=0.5
        )

@pytest.mark.parametrize("field", ["fake_probability", "confidence_score"])
@pytest.mark.parametrize("invalid_value", [-0.1, 1.1])
def test_image_prediction_out_of_bounds(field, invalid_value):
    data = {
        "filename": "img.png",
        "label": "FAKE",
        "fake_probability": 0.6,
        "confidence_score": 0.4
    }
    data[field] = invalid_value
    with pytest.raises(ValidationError):
        ImagePrediction(**data)

def test_batch_prediction_response_valid():
    pred = ImagePrediction(
        filename = "test.png",
        label = "FAKE",
        fake_probability=0.9,
        confidence_score=0.9
    )
    response = BatchPredictionResponse(predictions=[pred])
    assert len(response.predictions) == 1
    assert response.predictions[0].filename == "test.png"

def test_http_error_detail_valid():
    error = HTTPErrorDetail(detail="The model is not ready.")
    assert error.detail == "The model is not ready."