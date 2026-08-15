import asyncio
import json

import httpx

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
    assert parsed["alcohol_content"] == "45% Alc."
    assert parsed["net_contents"] == "750 mL"
    assert parsed["warning_statement"].endswith("during pregnancy.")


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
                json={"choices": [{"message": {"content": json_dumps(extracted)}}]},
                request=request,
            )

    def json_dumps(value):
        return json.dumps(value)

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://provider.example/v1/")
    monkeypatch.setenv("LLM_MODEL", "vision-test-model")
    monkeypatch.setattr(vision_client.httpx, "AsyncClient", FakeClient)

    result = asyncio.run(vision_client._call_provider(b"png-bytes"))

    assert result.model_dump() == extracted
    assert captured["timeout"] == 5.0
    assert captured["url"] == "https://provider.example/v1/chat/completions"
    assert captured["payload"]["model"] == "vision-test-model"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["headers"]["Authorization"] == "Bearer test-key"
