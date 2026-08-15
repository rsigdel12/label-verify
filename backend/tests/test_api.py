import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.extraction.errors import ExtractionUnavailableError
from app.extraction.schema import ExtractedLabel
from app.main import app

client = TestClient(app)
FIXTURES = Path(__file__).parent / "fixtures"


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


def test_readiness_reports_bundled_local_ocr():
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["extraction"]["local_ocr_available"] is True


def test_application_startup_preloads_local_ocr(monkeypatch):
    calls = []

    monkeypatch.setattr("app.main.initialize_local_ocr", lambda: calls.append("loaded"))
    with TestClient(app) as lifespan_client:
        assert lifespan_client.get("/health").status_code == 200

    assert calls == ["loaded"]


def test_verify_clean_fixture_end_to_end():
    metadata = json.loads(
        (FIXTURES / "fixture_01_clean_match.json").read_text(encoding="utf-8")
    )
    response = client.post(
        "/verify",
        files={
            "file": (
                "fixture_01_clean_match.png",
                (FIXTURES / "fixture_01_clean_match.png").read_bytes(),
                "image/png",
            )
        },
        data={"application_data": json.dumps(metadata["application_data"])},
    )

    assert response.status_code == 200
    assert response.json()["processing_time_ms"] < 5000
    assert {item["status"] for item in response.json()["comparison"].values()} == {
        "pass"
    }


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


def test_batch_verifies_multiple_real_images():
    metadata = json.loads(
        (FIXTURES / "fixture_01_clean_match.json").read_text(encoding="utf-8")
    )
    response = client.post(
        "/verify/batch",
        files=[
            (
                "files",
                (
                    name,
                    (FIXTURES / name).read_bytes(),
                    "image/png",
                ),
            )
            for name in (
                "fixture_01_clean_match.png",
                "fixture_02_brand_case_variation.png",
            )
        ],
        data={"application_data": json.dumps(metadata["application_data"])},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["progress"]["status"] == "complete"
    assert payload["progress"]["completed"] == 2
    assert all(
        item["status"] == "pass"
        for result in payload["results"]
        for item in result["comparison"].values()
    )
