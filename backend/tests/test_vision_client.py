import asyncio
import json
from pathlib import Path

import httpx
import pytest

from app.extraction import vision_client


def test_ocr_parser_keeps_multiline_warning_together():
    text = """ACME SPIRITS
Kentucky Straight Bourbon Whiskey
45% Alc. by Vol.
750 mL
GOVERNMENT WARNING: (1) According to the Surgeon General,
women should not drink alcoholic beverages during pregnancy.
"""

    parsed = vision_client._extract_from_text(text)

    assert parsed["brand_name"] == "ACME SPIRITS"
    assert parsed["class_type"] == "Kentucky Straight Bourbon Whiskey"
    assert parsed["alcohol_content"] == "45% Alc. by Vol."
    assert parsed["net_contents"] == "750 mL"
    assert parsed["warning_statement"].endswith("during pregnancy.")


def test_ocr_parser_joins_wrapped_class_and_excludes_batch_code():
    text = """Acme Spirits
KENTUCKY STRAIGHT BOURBON
WHISKEY
45% Alc. by Vol.
NET CONTENTS 750 mL
GOVERNMENT WARNING: Exact warning.
BATCH 26-0815
"""

    parsed = vision_client._extract_from_text(text)

    assert parsed["class_type"] == "KENTUCKY STRAIGHT BOURBON WHISKEY"
    assert parsed["warning_statement"] == "GOVERNMENT WARNING: Exact warning."


def test_ocr_parser_uses_visual_hierarchy_and_keeps_ocr_typo_in_class():
    text = """SMALL RELEASE
Acme Spirits
KENTUCKY STRAIGHT BOURBAN
WHISKEY
ALC. 45% BY VOL.
750 ML
GOVERNMENT WARNING: Exact warning.
"""

    parsed = vision_client._extract_from_text(
        text,
        line_heights=[14, 42, 25, 25, 18, 18, 14],
    )

    assert parsed["brand_name"] == "Acme Spirits"
    assert parsed["class_type"] == "KENTUCKY STRAIGHT BOURBAN WHISKEY"
    assert parsed["alcohol_content"] == "ALC. 45% BY VOL."
    assert parsed["net_contents"] == "750 ML"


def test_ocr_parser_handles_split_abv_centiliters_and_noisy_warning_heading():
    text = """NORTH COAST
LONDON DRY GIN
ALCOHOL
40,0 PERCENT BY VOLUME
NET CONT. 75 cL
G0VERNMENT WARN1NG:
(1) According to the Surgeon General, women should not drink alcoholic beverages
during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic
beverages impairs your ability to drive a car or operate machinery, and may cause health problems.
LOT 123
"""

    parsed = vision_client._extract_from_text(text)

    assert parsed["class_type"] == "LONDON DRY GIN"
    assert parsed["alcohol_content"] == "40,0 PERCENT BY VOLUME"
    assert parsed["net_contents"] == "75 cL"
    assert parsed["warning_statement"].startswith("G0VERNMENT WARN1NG:")
    assert parsed["warning_statement"].endswith("health problems.")


def test_ocr_parser_handles_proof_and_fluid_ounces():
    parsed = vision_client._extract_from_text(
        "Example Brand\nRYE WHISKEY\n80 PROOF\nNET CONTENTS 12 FL. OZ."
    )

    assert parsed["class_type"] == "RYE WHISKEY"
    assert parsed["alcohol_content"] == "80 PROOF"
    assert parsed["net_contents"] == "12 FL. OZ"


def test_bundled_local_ocr_reads_clean_fixture(monkeypatch):
    fixture = Path(__file__).parent / "fixtures" / "fixture_01_clean_match.png"
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    extracted = asyncio.run(vision_client.extract_label_fields(fixture.read_bytes()))

    assert extracted.brand_name == "Acme Spirits"
    assert extracted.class_type == "KENTUCKY STRAIGHT BOURBON WHISKEY"
    assert extracted.alcohol_content == "45% Alc. by Vol."
    assert extracted.net_contents == "750 mL"
    assert extracted.warning_statement.endswith("may cause health problems.")


def test_complete_vision_result_skips_local_ocr(monkeypatch):
    complete = {
        "brand_name": "Acme Spirits",
        "class_type": "Vodka",
        "alcohol_content": "40% Alc. by Vol.",
        "net_contents": "750 mL",
        "warning_statement": (
            "GOVERNMENT WARNING: (1) According to the Surgeon General, women should "
            "not drink alcoholic beverages during pregnancy because of the risk of "
            "birth defects. (2) Consumption of alcoholic beverages impairs your ability "
            "to drive a car or operate machinery, and may cause health problems."
        ),
    }

    async def complete_provider(_image, _timeout):
        return vision_client.ExtractedLabel(**complete)

    monkeypatch.setenv("OPENAI_API_KEY", "configured")
    monkeypatch.setattr(
        vision_client, "_normalize_image", lambda image, **_kwargs: image
    )
    monkeypatch.setattr(vision_client, "rapidocr_available", lambda: True)
    monkeypatch.setattr(
        vision_client,
        "_run_rapidocr",
        lambda _image: (_ for _ in ()).throw(
            AssertionError("local OCR should not run for a complete vision result")
        ),
    )
    monkeypatch.setattr(vision_client, "_call_provider", complete_provider)

    result = asyncio.run(vision_client.extract_label_fields(b"image", "vision"))

    assert result.model_dump() == complete


def test_failed_vision_attempt_falls_back_to_local_ocr(monkeypatch):
    complete = {
        "brand_name": "Acme Spirits",
        "class_type": "Vodka",
        "alcohol_content": "40% Alc. by Vol.",
        "net_contents": "750 mL",
        "warning_statement": (
            "GOVERNMENT WARNING: (1) According to the Surgeon General, women should "
            "not drink alcoholic beverages during pregnancy because of the risk of "
            "birth defects. (2) Consumption of alcoholic beverages impairs your ability "
            "to drive a car or operate machinery, and may cause health problems."
        ),
    }

    async def failed_provider(_image, _timeout):
        return None

    monkeypatch.setenv("OPENAI_API_KEY", "configured")
    monkeypatch.setattr(
        vision_client, "_normalize_image", lambda image, **_kwargs: image
    )
    monkeypatch.setattr(vision_client, "rapidocr_available", lambda: True)
    monkeypatch.setattr(vision_client, "_run_rapidocr", lambda _image: complete)
    monkeypatch.setattr(vision_client, "_call_provider", failed_provider)

    result = asyncio.run(vision_client.extract_label_fields(b"image", "auto"))

    assert result.model_dump() == complete


def test_vision_mode_fails_fast_instead_of_running_slow_local_fallback(monkeypatch):
    async def failed_provider(_image, timeout):
        assert timeout == vision_client.ACCURATE_TIMEOUT_SECONDS
        return None

    monkeypatch.setenv("OPENAI_API_KEY", "configured")
    monkeypatch.setattr(
        vision_client, "_normalize_image", lambda image, **_kwargs: image
    )
    monkeypatch.setattr(vision_client, "_call_provider", failed_provider)
    monkeypatch.setattr(
        vision_client,
        "_run_rapidocr",
        lambda _image: (_ for _ in ()).throw(
            AssertionError("vision mode must not run the slow fallback")
        ),
    )

    with pytest.raises(vision_client.ExtractionUnavailableError, match="15 seconds"):
        asyncio.run(vision_client.extract_label_fields(b"image", "vision"))


def test_accurate_mode_requires_a_configured_provider(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(
        vision_client.ExtractionUnavailableError, match="Choose Fast read"
    ):
        asyncio.run(vision_client.extract_label_fields(b"image", "vision"))


def test_provider_request_parses_structured_label(monkeypatch):
    extracted = {
        "brand_name": "Acme Spirits",
        "class_type": "Vodka",
        "alcohol_content": "40% Alc. by Vol.",
        "net_contents": "750 mL",
        "warning_statement": "GOVERNMENT WARNING: Exact warning.",
    }
    captured = {}

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, json, headers):
            captured.update(url=url, payload=json, headers=headers)
            request = httpx.Request("POST", url)
            return httpx.Response(
                200,
                json={
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": json_dumps(extracted)}
                            ],
                        }
                    ]
                },
                request=request,
            )

    def json_dumps(value):
        return json.dumps(value)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("VISION_BASE_URL", "https://provider.example/v1/")
    monkeypatch.setenv("VISION_MODEL", "vision-test-model")
    monkeypatch.setattr(vision_client.httpx, "AsyncClient", FakeClient)

    result = asyncio.run(vision_client._call_provider(b"png-bytes"))

    assert result.model_dump() == extracted
    assert captured["timeout"] == vision_client.ACCURATE_TIMEOUT_SECONDS
    assert captured["url"] == "https://provider.example/v1/responses"
    assert captured["payload"]["model"] == "vision-test-model"
    assert captured["payload"]["input"][0]["content"][1]["type"] == "input_image"
    assert captured["payload"]["input"][0]["content"][1]["detail"] == "original"
    assert captured["payload"]["text"]["format"]["type"] == "json_schema"
    assert captured["payload"]["text"]["format"]["strict"] is True
    assert captured["payload"]["reasoning"] == {"effort": "none"}
    assert captured["headers"]["Authorization"] == "Bearer test-key"


def test_free_gemini_provider_sends_image_and_parses_schema(monkeypatch):
    extracted = {
        "brand_name": "Acme Spirits",
        "class_type": "Vodka",
        "alcohol_content": "40% Alc. by Vol.",
        "net_contents": "750 mL",
        "warning_statement": "GOVERNMENT WARNING: Exact warning.",
    }
    captured = {}
    serialized = json.dumps(extracted)

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, json, headers):
            captured.update(url=url, payload=json, headers=headers)
            request = httpx.Request("POST", url)
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": serialized}]
                            }
                        }
                    ]
                },
                request=request,
            )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "free-test-key")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://gemini.example/v1beta/")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    monkeypatch.setattr(vision_client.httpx, "AsyncClient", FakeClient)

    result = asyncio.run(vision_client._call_provider(b"jpeg-bytes"))

    assert result.model_dump() == extracted
    assert captured["timeout"] == vision_client.ACCURATE_TIMEOUT_SECONDS
    assert captured["url"] == (
        "https://gemini.example/v1beta/models/gemini-test-model:generateContent"
    )
    assert captured["headers"]["x-goog-api-key"] == "free-test-key"
    assert captured["payload"]["contents"][0]["parts"][1]["inlineData"] == {
        "mimeType": "image/jpeg",
        "data": "anBlZy1ieXRlcw==",
    }
    config = captured["payload"]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == vision_client.LABEL_JSON_SCHEMA
    assert config["thinkingConfig"] == {"thinkingLevel": "LOW"}
