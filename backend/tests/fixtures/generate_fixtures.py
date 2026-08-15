import json
import textwrap
from copy import deepcopy
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

BASE_DIR = Path(__file__).resolve().parent

STANDARD_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not "
    "drink alcoholic beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
    "operate machinery, and may cause health problems."
)

BASE_APPLICATION = {
    "brand_name": "Acme Spirits",
    "class_type": "Kentucky Straight Bourbon Whiskey",
    "alcohol_content": "45% Alc. by Vol.",
    "net_contents": "750 mL",
    "warning_statement": STANDARD_WARNING,
}


def font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def draw_label(img_path: Path, label_data: dict, skew=0.0, glare=0.0):
    image = Image.new("RGB", (1200, 1600), color=(236, 230, 214))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (70, 45, 1130, 1555), radius=35, fill=(252, 249, 239),
        outline=(153, 112, 44), width=8,
    )
    draw.rectangle((103, 78, 1097, 1522), outline=(25, 45, 62), width=3)
    draw.text(
        (600, 165), label_data["brand_name"], fill=(25, 45, 62),
        font=font(72, bold=True), anchor="mm",
    )
    draw.line((220, 235, 980, 235), fill=(153, 112, 44), width=4)
    draw.multiline_text(
        (600, 350),
        "\n".join(textwrap.wrap(label_data["class_type"].upper(), width=27)),
        fill=(45, 40, 34), font=font(48, bold=True), anchor="mm",
        align="center", spacing=12,
    )
    draw.text(
        (600, 530), label_data["alcohol_content"], fill=(25, 45, 62),
        font=font(42, bold=True), anchor="mm",
    )
    draw.text(
        (600, 615), f"NET CONTENTS {label_data['net_contents']}",
        fill=(25, 45, 62), font=font(36), anchor="mm",
    )
    draw.text(
        (600, 740),
        "DISTILLED AND BOTTLED BY ACME SPIRITS\nLOUISVILLE, KENTUCKY",
        fill=(65, 59, 51), font=font(28), anchor="mm", align="center", spacing=8,
    )

    warning = label_data["warning_statement"]
    prefix, separator, remainder = warning.partition(":")
    heading = f"{prefix}:" if separator else prefix
    draw.rounded_rectangle(
        (115, 880, 1085, 1375), radius=16, fill=(244, 239, 224),
        outline=(65, 59, 51), width=3,
    )
    draw.text(
        (150, 925), heading, fill=(30, 30, 30), font=font(31, bold=True)
    )
    draw.multiline_text(
        (150, 980), "\n".join(textwrap.wrap(remainder.strip(), width=61)),
        fill=(30, 30, 30), font=font(29), spacing=13,
    )
    draw.text(
        (600, 1450), "BATCH 26-0815", fill=(100, 91, 79),
        font=font(24), anchor="mm",
    )

    # Apply image degradation after drawing so it affects the text under test.
    if skew:
        image = image.rotate(
            skew, resample=Image.Resampling.BICUBIC, expand=False,
            fillcolor=(255, 255, 255),
        )
    if glare:
        glare_layer = Image.new("RGBA", image.size, (255, 255, 255, 0))
        glare_draw = ImageDraw.Draw(glare_layer)
        glare_draw.polygon(
            [(650, 0), (980, 0), (620, 1600), (280, 1600)],
            fill=(255, 255, 255, int(glare * 255)),
        )
        image = Image.alpha_composite(image.convert("RGBA"), glare_layer).convert("RGB")
        image = ImageEnhance.Contrast(image).enhance(0.92)

    image.save(img_path, optimize=True)


def fixture(name, label_changes=None, app_changes=None, statuses=None, **effects):
    label_data = deepcopy(BASE_APPLICATION)
    application_data = deepcopy(BASE_APPLICATION)
    label_data.update(label_changes or {})
    application_data.update(app_changes or {})
    expected_statuses = {field: "pass" for field in BASE_APPLICATION}
    expected_statuses.update(statuses or {})
    return {
        "name": name,
        "label_data": label_data,
        "application_data": application_data,
        "expected_statuses": expected_statuses,
        **effects,
    }


fixtures = [
    fixture("fixture_01_clean_match"),
    fixture(
        "fixture_02_brand_case_variation",
        label_changes={"brand_name": "ACME SPIRITS"},
    ),
    fixture(
        "fixture_03_wrong_warning_case",
        label_changes={
            "warning_statement": STANDARD_WARNING.replace(
                "GOVERNMENT WARNING:", "Government Warning:"
            )
        },
        statuses={"warning_statement": "fail"},
    ),
    fixture(
        "fixture_04_abv_mismatch",
        label_changes={"alcohol_content": "43% Alc. by Vol."},
        statuses={"alcohol_content": "fail"},
    ),
    fixture(
        "fixture_05_class_type_fuzzy",
        label_changes={"class_type": "Kentucky Straight Bourban Whiskey"},
        statuses={"class_type": "needs_review"},
    ),
    fixture(
        "fixture_06_net_contents_mismatch",
        label_changes={"net_contents": "1 L"},
        statuses={"net_contents": "fail"},
    ),
    fixture("fixture_07_skewed_label", skew=7),
    fixture("fixture_08_glare_label", glare=0.42),
]


def main():
    for item in fixtures:
        png_path = BASE_DIR / f"{item['name']}.png"
        json_path = BASE_DIR / f"{item['name']}.json"
        draw_label(
            png_path,
            item["label_data"],
            skew=item.get("skew", 0),
            glare=item.get("glare", 0),
        )
        metadata = {
            "label_data": item["label_data"],
            "application_data": item["application_data"],
            "expected_statuses": item["expected_statuses"],
        }
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Generated {len(fixtures)} fixtures in {BASE_DIR}")


if __name__ == "__main__":
    main()
