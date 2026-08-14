from rapidfuzz import fuzz

from app.comparison.rules import FUZZY_MATCH_THRESHOLD
from app.extraction.schema import ExtractedLabel


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split()).lower()


def _compare_fuzzy(expected: str | None, actual: str | None) -> dict:
    if expected is None or actual is None:
        return {"status": "needs_review", "expected": expected, "actual": actual}

    normalized_expected = _normalize_text(expected)
    normalized_actual = _normalize_text(actual)
    score = fuzz.ratio(normalized_expected, normalized_actual)

    return {
        "status": "pass" if score >= FUZZY_MATCH_THRESHOLD else "fail",
        "expected": expected,
        "actual": actual,
    }


def _compare_exact(expected: str | None, actual: str | None) -> dict:
    if expected is None or actual is None:
        return {"status": "needs_review", "expected": expected, "actual": actual}

    return {
        "status": "pass" if str(expected).strip() == str(actual).strip() else "fail",
        "expected": expected,
        "actual": actual,
    }


def _compare_warning_statement(expected: str | None, actual: str | None) -> dict:
    if expected is None or actual is None:
        return {"status": "needs_review", "expected": expected, "actual": actual}

    return {
        "status": (
            "pass" if expected.encode("utf-8") == actual.encode("utf-8") else "fail"
        ),
        "expected": expected,
        "actual": actual,
    }


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
        expected = getattr(extracted, field, None)
        actual = submitted.get(field)

        if field in {"brand_name", "class_type"}:
            results[field] = _compare_fuzzy(expected, actual)
        elif field == "warning_statement":
            results[field] = _compare_warning_statement(expected, actual)
        else:
            results[field] = _compare_exact(expected, actual)

    return results
