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
        "extractionMode",
        "modeHelp",
    } <= parser.ids


def test_frontend_submits_user_values_instead_of_a_fixed_verification_payload():
    script = (FRONTEND / "upload.js").read_text(encoding="utf-8")

    assert "function applicationPayload()" in script
    assert 'brand_name:byId("brandName").value.trim()' in script
    assert 'requestBody.append("application_data",JSON.stringify(applicationPayload()))' in script
    assert 'requestBody.append("extraction_mode",extractionMode.value)' in script
    assert "data.processing_time_ms" in script


def test_frontend_sample_and_batch_workflows_are_connected():
    script = (FRONTEND / "upload.js").read_text(encoding="utf-8")
    fixture = Path(__file__).parent / "fixtures" / "fixture_01_clean_match.png"

    assert (FRONTEND / "assets" / "sample-label.png").stat().st_size > 0
    assert (FRONTEND / "assets" / "sample-label.png").read_bytes() == fixture.read_bytes()
    assert 'fetch("./assets/sample-label.png?v=4")' in script
    assert 'fetch("/ready"' in script
    assert 'requestBody.append("files",file)' in script
    assert 'isBatch?"/verify/batch":"/verify"' in script


def test_standard_government_warning_is_prefilled_but_editable():
    markup = (FRONTEND / "index.html").read_text(encoding="utf-8")
    script = (FRONTEND / "upload.js").read_text(encoding="utf-8")

    assert '<textarea id="warningStatement"' in markup
    assert "readonly" not in markup
    assert 'byId("warningStatement").value = STANDARD_WARNING;' in script
    assert "According to the Surgeon General" in script


def test_class_type_has_common_enum_suggestions_without_blocking_specific_types():
    markup = (FRONTEND / "index.html").read_text(encoding="utf-8")

    assert 'list="alcoholTypes"' in markup
    assert '<datalist id="alcoholTypes">' in markup
    for value in ("Vodka", "Bourbon Whiskey", "Red Wine", "Lager", "Sake"):
        assert f'<option value="{value}">' in markup


def test_frontend_offers_fast_and_accurate_read_modes():
    markup = (FRONTEND / "index.html").read_text(encoding="utf-8")
    script = (FRONTEND / "upload.js").read_text(encoding="utf-8")

    assert '<option value="local">Fast' in markup
    assert '<option value="vision" disabled>Accurate' in markup
    assert "vision_provider_configured" in script
    assert "Accurate AI vision may take 5–15 seconds." in script
