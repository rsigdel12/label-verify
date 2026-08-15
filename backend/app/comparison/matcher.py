import re
from decimal import Decimal, InvalidOperation

from rapidfuzz import fuzz

from app.comparison.rules import FUZZY_MATCH_THRESHOLD
from app.extraction.schema import ExtractedLabel


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split()).lower()


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
    if normalized_expected == normalized_actual:
        status = "pass"
    elif score >= FUZZY_MATCH_THRESHOLD:
        # A close regulated designation may be an OCR error or a real label
        # typo. Do not silently approve it without a person checking the image.
        status = "needs_review"
    else:
        status = "fail"
    return _comparison(status, expected, actual, similarity=round(score, 1))


def _compare_exact(expected: str | None, actual: str | None) -> dict:
    if expected is None or actual is None:
        return _comparison("needs_review", expected, actual)

    status = "pass" if str(expected).strip() == str(actual).strip() else "fail"
    return _comparison(status, expected, actual)


def _decimal_match(pattern: str, value: str | None) -> Decimal | None:
    if value is None:
        return None
    match = re.search(pattern, str(value), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return Decimal(match.group(1))
    except InvalidOperation:
        return None


def _compare_alcohol_content(expected: str | None, actual: str | None) -> dict:
    if expected is None or actual is None:
        return _comparison("needs_review", expected, actual)

    expected_percent = _decimal_match(r"(\d+(?:\.\d+)?)\s*%", expected)
    actual_percent = _decimal_match(r"(\d+(?:\.\d+)?)\s*%", actual)
    if expected_percent is None or actual_percent is None:
        return _compare_exact(expected, actual)

    return _comparison(
        "pass" if expected_percent == actual_percent else "fail", expected, actual
    )


def _net_contents_ml(value: str | None) -> Decimal | None:
    if value is None:
        return None
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(ml|millilit(?:er|re)s?|l|lit(?:er|re)s?)\b",
        str(value),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    try:
        amount = Decimal(match.group(1))
    except InvalidOperation:
        return None
    unit = match.group(2).lower()
    return amount * 1000 if unit in {"l", "liter", "litre", "liters", "litres"} else amount


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

    # Line wrapping is a visual layout detail, not a wording difference. Casing,
    # punctuation, and all words remain exact after whitespace is collapsed.
    normalized_expected = " ".join(str(expected).strip().split())
    normalized_actual = " ".join(str(actual).strip().split())
    status = "pass" if normalized_expected == normalized_actual else "fail"
    return _comparison(status, expected, actual)


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
