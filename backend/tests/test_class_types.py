from app.extraction.class_types import (
    AlcoholType,
    COMMON_ALCOHOL_TYPE_VALUES,
    classify_alcohol_type,
    expected_type_accepts,
)


def test_common_alcohol_type_enum_values_are_unique():
    assert len(COMMON_ALCOHOL_TYPE_VALUES) == len(set(COMMON_ALCOHOL_TYPE_VALUES))
    assert {"Vodka", "Bourbon Whiskey", "Red Wine", "Lager", "Sake"} <= set(
        COMMON_ALCOHOL_TYPE_VALUES
    )


def test_classification_prefers_specific_types_and_tolerates_ocr_errors():
    assert classify_alcohol_type("KENTUCKY STRAIGHT BOURBON WHISKEY")[0] == (
        AlcoholType.BOURBON_WHISKEY
    )
    assert classify_alcohol_type("KENTUCKY STRAIGHT BOURBAN WHISKEY")[0] == (
        AlcoholType.BOURBON_WHISKEY
    )
    assert classify_alcohol_type("LONDON DRY GIN")[0] == AlcoholType.GIN


def test_broad_enum_accepts_subtype_but_not_a_different_subtype():
    assert expected_type_accepts(
        AlcoholType.WHISKEY, AlcoholType.BOURBON_WHISKEY
    )
    assert not expected_type_accepts(
        AlcoholType.BOURBON_WHISKEY, AlcoholType.RYE_WHISKEY
    )
