import re
from decimal import Decimal, InvalidOperation

from rapidfuzz import fuzz

from app.comparison.rules import FUZZY_MATCH_THRESHOLD
from app.extraction.class_types import (
    classify_alcohol_type,
    expected_type_accepts,
    same_type_family,
)
from app.extraction.schema import ExtractedLabel


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split()).casefold()


def _comparison(status: str, expected: str | None, actual: str | None, **details) -> dict:
    return {"status": status, "expected": expected, "actual": actual, **details}


def _compare_fuzzy(expected: str | None, actual: str | None) -> dict:
    if expected is None or actual is None:
        return _comparison("needs_review", expected, actual)

    normalized_expected = _normalize_text(expected)
    normalized_actual = _normalize_text(actual)
    score = fuzz.ratio(normalized_expected, normalized_actual)

    return _comparison(
        "pass" if score >= FUZZY_MATCH_THRESHOLD else "fail",
        expected,
        actual,
        similarity=round(score, 1),
    )


def _compare_class_type(expected: str | None, actual: str | None) -> dict:
    if expected is None or actual is None:
        return _comparison("needs_review", expected, actual)

    normalized_expected = _normalize_text(expected)
    normalized_actual = _normalize_text(actual)
    score = fuzz.ratio(normalized_expected, normalized_actual)
    expected_type, _ = classify_alcohol_type(expected)
    actual_type, _ = classify_alcohol_type(actual)
    if normalized_expected == normalized_actual:
        status = "pass"
    elif score >= FUZZY_MATCH_THRESHOLD:
        # A close regulated designation may be an OCR error or a real label
        # typo. Do not silently approve it without a person checking the image.
        status = "needs_review"
    elif expected_type is not None and expected_type == actual_type:
        # A reviewer may deliberately choose a common enum value such as
        # "Whiskey" while the label carries a more specific compliant
        # designation. That category match can pass. Two different detailed
        # strings in the same category still require a person to verify them.
        expected_is_enum_value = normalized_expected == _normalize_text(
            expected_type.value
        )
        status = "pass" if expected_is_enum_value else "needs_review"
    elif same_type_family(expected_type, actual_type):
        expected_is_enum_value = normalized_expected == _normalize_text(
            expected_type.value
        )
        if expected_is_enum_value and expected_type_accepts(expected_type, actual_type):
            status = "pass"
        elif expected_type_accepts(actual_type, expected_type):
            status = "needs_review"
        else:
            status = "fail"
    else:
        status = "fail"
    return _comparison(
        status,
        expected,
        actual,
        similarity=round(score, 1),
        matched_type=actual_type.value if actual_type else None,
    )


def _compare_exact(expected: str | None, actual: str | None) -> dict:
    if expected is None or actual is None:
        return _comparison("needs_review", expected, actual)

    status = "pass" if _normalize_text(expected) == _normalize_text(actual) else "fail"
    return _comparison(status, expected, actual)


def _decimal_match(pattern: str, value: str | None) -> Decimal | None:
    if value is None:
        return None
    match = re.search(pattern, str(value), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return Decimal(match.group(1).replace(",", "."))
    except InvalidOperation:
        return None


def _alcohol_percent(value: str | None) -> Decimal | None:
    if value is None:
        return None
    percent = _decimal_match(
        r"(\d+(?:[.,]\d+)?)\s*(?:%|percent\b|per\s*cent\b)", value
    )
    if percent is not None:
        return percent
    proof = _decimal_match(r"(\d+(?:[.,]\d+)?)\s*proof\b", value)
    return proof / 2 if proof is not None else None


def _compare_alcohol_content(expected: str | None, actual: str | None) -> dict:
    if expected is None or actual is None:
        return _comparison("needs_review", expected, actual)

    expected_percent = _alcohol_percent(expected)
    actual_percent = _alcohol_percent(actual)
    if expected_percent is None or actual_percent is None:
        return _compare_exact(expected, actual)

    return _comparison(
        "pass" if expected_percent == actual_percent else "fail", expected, actual
    )


def _net_contents_ml(value: str | None) -> Decimal | None:
    if value is None:
        return None
    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(m(?:l|1)?|rn(?:l|1)|millilit(?:er|re)s?|cl|centilit(?:er|re)s?|"
        r"l|lit(?:er|re)s?|fl\.?\s*oz\.?|fluid\s+ounces?|oz\.?)\b",
        str(value),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    try:
        amount = Decimal(match.group(1))
    except InvalidOperation:
        return None
    unit = re.sub(r"[.\s]", "", match.group(2).lower())
    if unit in {"l", "liter", "litre", "liters", "litres"}:
        return amount * 1000
    if unit in {"m1", "rnl", "rn1"}:
        return amount
    if unit in {"cl", "centiliter", "centilitre", "centiliters", "centilitres"}:
        return amount * 10
    if unit in {"floz", "fluidounce", "fluidounces", "oz"}:
        return amount * Decimal("29.5735295625")
    return amount


def _compare_net_contents(expected: str | None, actual: str | None) -> dict:
    if expected is None or actual is None:
        return _comparison("needs_review", expected, actual)

    expected_ml = _net_contents_ml(expected)
    actual_ml = _net_contents_ml(actual)
    if expected_ml is None or actual_ml is None:
        return _compare_exact(expected, actual)
    return _comparison("pass" if expected_ml == actual_ml else "fail", expected, actual)


def _compare_warning_statement(expected: str | None, actual: str | None) -> dict:
    if expected is None or actual is None:
        return _comparison("needs_review", expected, actual)

    # Whitespace and capitalization are presentation details for this
    # prototype. Spelling, punctuation, numbering, and word order still need
    # to match.
    normalized_expected = re.sub(r"\s+", "", str(expected).strip()).casefold()
    normalized_actual = re.sub(r"\s+", "", str(actual).strip()).casefold()
    if normalized_expected == normalized_actual:
        return _comparison("pass", expected, actual, similarity=100.0)

    similarity = fuzz.ratio(
        str(expected).strip().casefold(), str(actual).strip().casefold()
    )
    heading_verified = bool(
        re.match(r"^government\s+warning\s*:", str(actual).strip(), re.IGNORECASE)
    )
    # A near-complete OCR transcription is evidence that the warning is
    # present, but it is not safe to approve exact regulated wording when OCR
    # dropped punctuation or confused a character.
    status = "needs_review" if similarity >= 90 and heading_verified else "fail"
    return _comparison(
        status,
        expected,
        actual,
        similarity=round(similarity, 1),
        heading_verified=heading_verified,
    )


def compare_fields(extracted: ExtractedLabel, submitted: dict) -> dict:
    """Compare extracted label values to submitted application data."""
    results: dict[str, dict] = {}

    for field in [
        "brand_name",
        "class_type",
        "alcohol_content",
        "net_contents",
        "warning_statement",
    ]:
        # The application is the expected value; the image extraction is what
        # was actually printed on the label.
        expected = submitted.get(field)
        actual = getattr(extracted, field, None)

        if field == "brand_name":
            results[field] = _compare_fuzzy(expected, actual)
        elif field == "class_type":
            results[field] = _compare_class_type(expected, actual)
        elif field == "warning_statement":
            results[field] = _compare_warning_statement(expected, actual)
        elif field == "alcohol_content":
            results[field] = _compare_alcohol_content(expected, actual)
        elif field == "net_contents":
            results[field] = _compare_net_contents(expected, actual)
        else:
            results[field] = _compare_exact(expected, actual)

    return results
