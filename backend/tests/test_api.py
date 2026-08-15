import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from app.extraction.errors import ExtractionUnavailableError
from app.extraction.schema import ExtractedLabel
from app.main import app

client = TestClient(app)


def image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(output, format="PNG")
    return output.getvalue()


def submission() -> dict:
    return {
        "brand_name": "Acme Spirits",
        "class_type": "Vodka",
        "alcohol_content": "40% Alcohol by Volume",
        "net_contents": "750 mL",
        "warning_statement": "GOVERNMENT WARNING: Exact text.",
    }


def test_verify_success(monkeypatch):
    async def fake_extract(_image):
        return ExtractedLabel(**submission())

    monkeypatch.setattr("app.routes.verify.extract_label_fields", fake_extract)
    response = client.post(
        "/verify",
        files={"file": ("label.png", image_bytes(), "image/png")},
        data={"application_data": json.dumps(submission())},
    )

    assert response.status_code == 200
    assert response.json()["processing_time_ms"] >= 0
    assert all(
        result["status"] == "pass"
        for result in response.json()["comparison"].values()
    )


def test_verify_rejects_invalid_json():
    response = client.post(
        "/verify",
        files={"file": ("label.png", image_bytes(), "image/png")},
        data={"application_data": "not-json"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "application_data must be valid JSON."


def test_verify_rejects_non_image_upload():
    response = client.post(
        "/verify",
        files={"file": ("label.txt", b"hello", "text/plain")},
        data={"application_data": json.dumps(submission())},
    )

    assert response.status_code == 415


def test_verify_rejects_fake_image_bytes():
    response = client.post(
        "/verify",
        files={"file": ("label.png", b"not an image", "image/png")},
        data={"application_data": json.dumps(submission())},
    )

    assert response.status_code == 400
    assert "not a valid" in response.json()["detail"]


def test_verify_returns_actionable_error_when_extraction_is_unavailable(monkeypatch):
    async def unavailable(_image):
        raise ExtractionUnavailableError("Configure LLM_API_KEY or install Tesseract OCR.")

    monkeypatch.setattr("app.routes.verify.extract_label_fields", unavailable)
    response = client.post(
        "/verify",
        files={"file": ("label.png", image_bytes(), "image/png")},
        data={"application_data": json.dumps(submission())},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Configure LLM_API_KEY or install Tesseract OCR."


def test_batch_requires_one_application_per_file_when_given_a_list():
    response = client.post(
        "/verify/batch",
        files=[
            ("files", ("one.png", image_bytes(), "image/png")),
            ("files", ("two.png", image_bytes(), "image/png")),
        ],
        data={"application_data": json.dumps([submission()])},
    )

    assert response.status_code == 400


def test_batch_reports_per_file_errors(monkeypatch):
    async def fake_extract(_image):
        return ExtractedLabel(**submission())

    monkeypatch.setattr("app.routes.batch.extract_label_fields", fake_extract)
    response = client.post(
        "/verify/batch",
        files=[
            ("files", ("good.png", image_bytes(), "image/png")),
            ("files", ("bad.txt", b"hello", "text/plain")),
        ],
        data={"application_data": json.dumps(submission())},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["progress"]["total"] == 2
    assert payload["progress"]["completed"] == 1
    assert payload["progress"]["failed"] == 1
    assert payload["progress"]["status"] == "complete_with_errors"
    assert payload["progress"]["processing_time_ms"] >= 0
    assert payload["results"][1]["error"]["status"] == 415
    assert payload["results"][1]["processing_time_ms"] >= 0
