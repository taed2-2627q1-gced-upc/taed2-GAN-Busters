from unittest.mock import MagicMock, patch
import pytest
from contextlib import ExitStack
from fastapi import status
from fastapi.testclient import TestClient

from gan_busters.api.app import app, lifespan


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"message": "Welcome to the gan-busters app"}


def test_liveness_endpoint(client):
    response = client.get("/health/live")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "alive"}


def test_readiness_model_not_ready(client):
    with patch("gan_busters.api.app.model", None):
        response = client.get("/health/ready")
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "not ready" in response.json()["detail"].lower()


def test_readiness_model_ready(client):
    with patch("gan_busters.api.app.model", MagicMock()):
        response = client.get("/health/ready")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True


def test_predict_no_files(client):
    with patch("gan_busters.api.app.model", MagicMock()):
        response = client.post("/predict", files=[])
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_predict_invalid_extension(client, tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("hello")

    with patch("gan_busters.api.app.model", MagicMock()):
        with open(txt_file, "rb") as f:
            response = client.post(
                "/predict",
                files={"files": ("test.txt", f, "text/plain")}
            )
        assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

def test_predict_unreadable_image(client, tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b'\xC3\xA9')

    with patch("gan_busters.api.app.model", MagicMock()):
        with open(image, "rb") as f:
            response = client.post(
                "/predict",
                files={"files": ("image.jpg", f, "image/jpeg")}
            )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

def test_predict_success(client, sample_image):
    mock_model = MagicMock()
    mock_results = [{
        "label": "REAL",
        "fake_probability": 0.1,
        "confidence_score": 0.9
    }]

    with patch("gan_busters.api.app.model", mock_model), \
         patch("gan_busters.api.app.predict_images", return_value=mock_results):
        
        with open(sample_image, "rb") as f:
            response = client.post(
                "/predict",
                files={"files": ("sample.png", f, "image/png")}
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["label"] == "REAL"


def test_analytics_performance(client):
    with patch("gan_busters.api.app.get_snapshot", return_value={"requests": {"total_recorded": 5}}):
        response = client.get("/analytics/performance")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "api" in data
        assert "model" in data
        assert data["api"]["requests"]["total_recorded"] == 5

def test_maximum_amount_files(client, tmp_path):
    images = [tmp_path / f"image_{i}.jpg" for i in range(11)]
    for img in images:
        img.write_bytes(b'\xC3\xA9')

    with patch("gan_busters.api.app.model", MagicMock()):
        with ExitStack() as stack:
            opened_files = [stack.enter_context(open(img, "rb")) for img in images]
            file_payload = [
                ("files", (img.name, f, "image/jpeg"))
                for img, f in zip(images, opened_files)
            ]
            response = client.post("/predict", files=file_payload)
            print(response.json())
        assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE