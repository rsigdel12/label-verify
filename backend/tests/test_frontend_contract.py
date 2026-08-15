from html.parser import HTMLParser
from pathlib import Path


FRONTEND = Path(__file__).parents[2] / "frontend"


class FormMarkupParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.application_fields = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if element_id := attributes.get("id"):
            self.ids.add(element_id)
        if tag in {"input", "textarea"} and (name := attributes.get("name")):
            self.application_fields.add(name)


def test_frontend_exposes_application_inputs_upload_and_timing_results():
    parser = FormMarkupParser()
    parser.feed((FRONTEND / "index.html").read_text(encoding="utf-8"))

    assert parser.application_fields == {
        "brand_name",
        "class_type",
        "alcohol_content",
        "net_contents",
        "warning_statement",
    }
    assert {
        "verifyForm",
        "fileInput",
        "filePreview",
        "submitButton",
        "loadingElapsed",
        "resultsSection",
        "serverTime",
        "roundTripTime",
    } <= parser.ids


def test_frontend_submits_user_values_instead_of_a_fixed_verification_payload():
    script = (FRONTEND / "upload.js").read_text(encoding="utf-8")

    assert "function applicationPayload()" in script
    assert 'document.getElementById("brandName").value.trim()' in script
    assert 'requestBody.append("application_data", JSON.stringify(applicationPayload()))' in script
    assert "data.processing_time_ms" in script
