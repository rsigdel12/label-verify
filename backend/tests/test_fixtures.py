import json
from pathlib import Path

from PIL import Image

from app.comparison.matcher import compare_fields
from app.extraction.schema import ExtractedLabel

FIXTURES = Path(__file__).parent / "fixtures"


def test_fixture_metadata_and_expected_comparisons_are_consistent():
    metadata_files = sorted(FIXTURES.glob("fixture_*.json"))
    assert len(metadata_files) == 8

    for metadata_path in metadata_files:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert set(metadata) == {
            "label_data",
            "application_data",
            "expected_statuses",
        }
        extracted = ExtractedLabel(**metadata["label_data"])
        results = compare_fields(extracted, metadata["application_data"])
        statuses = {field: result["status"] for field, result in results.items()}
        assert statuses == metadata["expected_statuses"], metadata_path.name


def test_fixture_images_are_nonempty_and_have_expected_dimensions():
    for image_path in sorted(FIXTURES.glob("fixture_*.png")):
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            assert image.size == (1200, 1600)
