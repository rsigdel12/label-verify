import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent


def draw_label(img_path: Path, payload: dict, skew: float = 0.0, glare: float = 0.0):
    image = Image.new("RGB", (1200, 1600), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # background tint
    for i in range(0, 1600, 20):
        shade = 245 - (i // 30)
        draw.rectangle([0, i, 1200, i + 20], fill=(shade, shade, shade))

    if skew:
        image = image.rotate(skew, expand=True, fillcolor=(255, 255, 255))

    if glare:
        glare_layer = Image.new("RGBA", image.size, (255, 255, 255, 0))
        glare_draw = ImageDraw.Draw(glare_layer)
        glare_draw.rectangle(
            (100, 0, 1100, 1600), fill=(255, 255, 255, int(glare * 255))
        )
        image = Image.alpha_composite(image.convert("RGBA"), glare_layer).convert("RGB")

    draw = ImageDraw.Draw(image)
    font_big = ImageFont.load_default()
    font_med = ImageFont.load_default()
    font_small = ImageFont.load_default()

    draw.text((120, 110), payload["brand_name"], fill=(30, 30, 30), font=font_big)
    draw.text((120, 240), payload["class_type"], fill=(80, 80, 80), font=font_med)
    draw.text(
        (120, 350),
        f"Alcohol: {payload['alcohol_content']}",
        fill=(30, 30, 30),
        font=font_med,
    )
    draw.text(
        (120, 450),
        f"Net Contents: {payload['net_contents']}",
        fill=(30, 30, 30),
        font=font_med,
    )

    warning_box = (120, 560, 1080, 760)
    draw.rounded_rectangle(
        warning_box, radius=20, fill=(245, 240, 230), outline=(180, 180, 180), width=2
    )
    draw.text(
        (150, 600), payload["warning_statement"], fill=(60, 60, 60), font=font_small
    )

    image.save(img_path)


fixtures = [
    {
        "name": "fixture_01_clean_match",
        "payload": {
            "brand_name": "Acme Spirits",
            "class_type": "Vodka",
            "alcohol_content": "40% vol",
            "net_contents": "750 ml",
            "warning_statement": "Contains sulfites.",
        },
        "skew": 0,
        "glare": 0,
    },
    {
        "name": "fixture_02_brand_case_variation",
        "payload": {
            "brand_name": "ACME SPIRITS",
            "class_type": "Vodka",
            "alcohol_content": "40% vol",
            "net_contents": "750 ml",
            "warning_statement": "Contains sulfites.",
        },
        "skew": 0,
        "glare": 0,
    },
    {
        "name": "fixture_03_wrong_warning_case",
        "payload": {
            "brand_name": "Acme Spirits",
            "class_type": "Vodka",
            "alcohol_content": "40% vol",
            "net_contents": "750 ml",
            "warning_statement": "contains sulfites.",
        },
        "skew": 0,
        "glare": 0,
    },
    {
        "name": "fixture_04_abv_mismatch",
        "payload": {
            "brand_name": "Acme Spirits",
            "class_type": "Vodka",
            "alcohol_content": "38% vol",
            "net_contents": "750 ml",
            "warning_statement": "Contains sulfites.",
        },
        "skew": 0,
        "glare": 0,
    },
    {
        "name": "fixture_05_class_type_fuzzy",
        "payload": {
            "brand_name": "Acme Spirits",
            "class_type": "Vodka Premium",
            "alcohol_content": "40% vol",
            "net_contents": "750 ml",
            "warning_statement": "Contains sulfites.",
        },
        "skew": 0,
        "glare": 0,
    },
    {
        "name": "fixture_06_net_contents_mismatch",
        "payload": {
            "brand_name": "Acme Spirits",
            "class_type": "Vodka",
            "alcohol_content": "40% vol",
            "net_contents": "1 L",
            "warning_statement": "Contains sulfites.",
        },
        "skew": 0,
        "glare": 0,
    },
    {
        "name": "fixture_07_skewed_label",
        "payload": {
            "brand_name": "Acme Spirits",
            "class_type": "Vodka",
            "alcohol_content": "40% vol",
            "net_contents": "750 ml",
            "warning_statement": "Contains sulfites.",
        },
        "skew": 7,
        "glare": 0,
    },
    {
        "name": "fixture_08_glare_label",
        "payload": {
            "brand_name": "Acme Spirits",
            "class_type": "Vodka",
            "alcohol_content": "40% vol",
            "net_contents": "750 ml",
            "warning_statement": "Contains sulfites.",
        },
        "skew": 0,
        "glare": 0.35,
    },
]

for fixture in fixtures:
    png_path = BASE_DIR / f"{fixture['name']}.png"
    json_path = BASE_DIR / f"{fixture['name']}.json"
    draw_label(
        png_path, fixture["payload"], skew=fixture["skew"], glare=fixture["glare"]
    )
    json_path.write_text(json.dumps(fixture["payload"], indent=2), encoding="utf-8")

print(f"Generated {len(fixtures)} fixtures in {BASE_DIR}")
