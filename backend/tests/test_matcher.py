from app.comparison.matcher import compare_fields
from app.extraction.schema import ExtractedLabel


def test_clean_exact_match():
    extracted = ExtractedLabel(
        brand_name="Acme Spirits",
        class_type="Vodka",
        alcohol_content="40% vol",
        net_contents="750 ml",
        warning_statement="Contains sulfites.",
    )
    submitted = {
        "brand_name": "Acme Spirits",
        "class_type": "Vodka",
        "alcohol_content": "40% vol",
        "net_contents": "750 ml",
        "warning_statement": "Contains sulfites.",
    }

    results = compare_fields(extracted, submitted)

    assert results["brand_name"]["status"] == "pass"
    assert results["class_type"]["status"] == "pass"
    assert results["alcohol_content"]["status"] == "pass"
    assert results["net_contents"]["status"] == "pass"
    assert results["warning_statement"]["status"] == "pass"


def test_brand_name_case_difference_should_pass_fuzzy_matching():
    extracted = ExtractedLabel(brand_name="ACME SPIRITS")
    submitted = {"brand_name": "Acme   Spirits"}

    result = compare_fields(extracted, submitted)

    assert result["brand_name"]["status"] == "pass"


def test_warning_statement_wrong_casing_should_fail():
    extracted = ExtractedLabel(warning_statement="Contains Sulfites.")
    submitted = {"warning_statement": "contains sulfites."}

    result = compare_fields(extracted, submitted)

    assert result["warning_statement"]["status"] == "fail"


def test_different_abv_should_fail():
    extracted = ExtractedLabel(alcohol_content="40% vol")
    submitted = {"alcohol_content": "38% vol"}

    result = compare_fields(extracted, submitted)

    assert result["alcohol_content"]["status"] == "fail"
