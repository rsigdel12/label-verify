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


def test_expected_is_application_and_actual_is_extracted_label():
    extracted = ExtractedLabel(alcohol_content="38% Alc. by Vol.")
    submitted = {"alcohol_content": "40% Alcohol by Volume"}

    result = compare_fields(extracted, submitted)["alcohol_content"]

    assert result == {
        "status": "fail",
        "expected": "40% Alcohol by Volume",
        "actual": "38% Alc. by Vol.",
    }


def test_equivalent_abv_format_should_pass():
    extracted = ExtractedLabel(alcohol_content="45% Alc. by Vol. (90 Proof)")
    submitted = {"alcohol_content": "45% Alcohol By Volume"}

    result = compare_fields(extracted, submitted)

    assert result["alcohol_content"]["status"] == "pass"


def test_proof_and_decimal_comma_abv_should_match_percent():
    extracted = ExtractedLabel(alcohol_content="80 Proof")
    submitted = {"alcohol_content": "40,0% Alcohol by Volume"}

    result = compare_fields(extracted, submitted)

    assert result["alcohol_content"]["status"] == "pass"


def test_equivalent_metric_net_contents_should_pass():
    extracted = ExtractedLabel(net_contents="0.75 L")
    submitted = {"net_contents": "750 mL"}

    result = compare_fields(extracted, submitted)

    assert result["net_contents"]["status"] == "pass"


def test_centiliters_and_ocr_ml_unit_should_match_milliliters():
    assert compare_fields(
        ExtractedLabel(net_contents="75 cL"), {"net_contents": "750 mL"}
    )["net_contents"]["status"] == "pass"
    assert compare_fields(
        ExtractedLabel(net_contents="750 m1"), {"net_contents": "750 mL"}
    )["net_contents"]["status"] == "pass"


def test_warning_visual_line_wraps_do_not_create_false_failure():
    extracted = ExtractedLabel(
        warning_statement="GOVERNMENT WARNING: (1) According to the Surgeon\nGeneral."
    )
    submitted = {
        "warning_statement": "GOVERNMENT WARNING: (1) According to the Surgeon General."
    }

    result = compare_fields(extracted, submitted)

    assert result["warning_statement"]["status"] == "pass"


def test_warning_ocr_joined_words_do_not_create_false_failure():
    extracted = ExtractedLabel(
        warning_statement="GOVERNMENT WARNING: alcoholic beveragesduring pregnancy."
    )
    submitted = {
        "warning_statement": "GOVERNMENT WARNING: alcoholic beverages during pregnancy."
    }

    result = compare_fields(extracted, submitted)

    assert result["warning_statement"]["status"] == "pass"


def test_missing_extraction_requires_review():
    result = compare_fields(ExtractedLabel(), {"brand_name": "Acme Spirits"})

    assert result["brand_name"]["status"] == "needs_review"


def test_close_but_nonidentical_class_type_requires_review():
    extracted = ExtractedLabel(class_type="Kentucky Straight Bourban Whiskey")
    submitted = {"class_type": "Kentucky Straight Bourbon Whiskey"}

    result = compare_fields(extracted, submitted)

    assert result["class_type"]["status"] == "needs_review"


def test_common_class_enum_matches_a_more_specific_detected_type():
    extracted = ExtractedLabel(class_type="Kentucky Straight Bourbon Whiskey")
    submitted = {"class_type": "Whiskey"}

    result = compare_fields(extracted, submitted)["class_type"]

    assert result["status"] == "pass"
    assert result["matched_type"] == "Bourbon Whiskey"


def test_different_whiskey_subtypes_do_not_pass_enum_matching():
    extracted = ExtractedLabel(class_type="Straight Rye Whiskey")
    submitted = {"class_type": "Bourbon Whiskey"}

    assert compare_fields(extracted, submitted)["class_type"]["status"] == "fail"


def test_near_complete_warning_requires_review_instead_of_false_failure():
    expected = (
        "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not "
        "drink alcoholic beverages during pregnancy because of the risk of birth defects."
    )
    actual = expected.replace("defects.", "defects")

    result = compare_fields(
        ExtractedLabel(warning_statement=actual), {"warning_statement": expected}
    )["warning_statement"]

    assert result["status"] == "needs_review"
    assert result["heading_verified"] is True
