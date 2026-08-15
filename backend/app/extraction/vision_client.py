import base64
import json
import logging
import os
import re
import shutil
from io import BytesIO
from typing import Optional

import httpx
from PIL import Image, UnidentifiedImageError

from app.extraction.errors import ExtractionUnavailableError, InvalidImageError
from app.extraction.schema import ExtractedLabel

logger = logging.getLogger(__name__)
MAX_IMAGE_PIXELS = 20_000_000


def _normalize_image(image_bytes: bytes) -> bytes:
    if not image_bytes:
        raise InvalidImageError("The uploaded image is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise InvalidImageError(
                    "The image dimensions are invalid or exceed 20 megapixels."
                )
            image = image.convert("RGB")
            image.thumbnail((2400, 2400))
            output = BytesIO()
            image.save(output, format="PNG", optimize=True)
            return output.getvalue()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(
            "The uploaded file is not a valid PNG, JPEG, WEBP, or GIF image."
        ) from exc


# Simple heuristics for parsing OCR text into fields
def _extract_from_text(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    joined = "\n".join(lines)

    # Brand: assume first non-empty line
    brand = lines[0] if lines else None

    # Class/type: look for known keywords or use second line
    class_type = None
    keywords = [
        "vodka",
        "whiskey",
        "whisky",
        "bourbon",
        "rum",
        "gin",
        "wine",
        "beer",
        "whiskey",
        "bourbon",
    ]
    for ln in lines[:4]:
        if any(k in ln.lower() for k in keywords):
            class_type = ln
            break
    if not class_type and len(lines) > 1:
        class_type = lines[1]

    # Alcohol content (ABV)
    abv_match = re.search(
        r"(\d{1,3}(?:\.\d)?\s*%\s*(?:alc\.?|alc|vol|Alc\.?|vol\.?))",
        joined,
        re.IGNORECASE,
    )
    alcohol_content = abv_match.group(1) if abv_match else None

    # Net contents (ml, L)
    net_match = re.search(r"(\d+\s*(?:ml|mL|l|L))", joined)
    net_contents = net_match.group(1) if net_match else None

    # Warning statement: preserve the complete warning, even when OCR wraps it.
    warning = None
    for index, ln in enumerate(lines):
        if "warning" in ln.lower():
            warning = " ".join(lines[index:])
            break

    return {
        "brand_name": brand,
        "class_type": class_type,
        "alcohol_content": alcohol_content,
        "net_contents": net_contents,
        "warning_statement": warning,
    }


async def _call_provider(image_bytes: bytes) -> Optional[ExtractedLabel]:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{encoded_image}"

    prompt = (
        "Extract the following fields from this alcohol label and return valid JSON only. "
        "The JSON object must include exactly these keys with string values: "
        "brand_name, class_type, alcohol_content, net_contents, warning_statement. "
        "For warning_statement, transcribe the complete statement exactly, preserving "
        "capitalization, punctuation, and numbering while joining visual line wraps with "
        "single spaces. If a field is not visible, use null. Do not include extra keys."
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        logger.warning("Vision provider request failed: %s", type(exc).__name__)
        return None

    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    if not content:
        return None

    try:
        if isinstance(content, str):
            parsed = json.loads(content)
        elif isinstance(content, dict):
            parsed = content
        else:
            parsed = json.loads(str(content))
    except (TypeError, ValueError, json.JSONDecodeError):
        logger.warning("Vision provider returned malformed JSON.")
        return None

    normalized = {
        field: None if parsed.get(field) is None else str(parsed[field])
        for field in (
            "brand_name",
            "class_type",
            "alcohol_content",
            "net_contents",
            "warning_statement",
        )
    }

    return ExtractedLabel(**normalized)


async def extract_label_fields(image_bytes: bytes) -> ExtractedLabel:
    """Try provider extraction first (if key present); otherwise fall back to OCR.

    The OCR fallback uses `pytesseract` and simple heuristics to extract the
    five required fields so the app can run without paid LLM access.
    """
    normalized_image = _normalize_image(image_bytes)

    # 1) Try provider if configured.
    label = await _call_provider(normalized_image)
    if label is not None:
        return label

    # 2) Fallback to local OCR
    try:
        import pytesseract
    except ImportError as exc:
        raise ExtractionUnavailableError(
            "Image extraction is unavailable. Configure LLM_API_KEY or install Tesseract OCR."
        ) from exc

    if shutil.which("tesseract") is None:
        raise ExtractionUnavailableError(
            "Image extraction is unavailable. Configure a working LLM_API_KEY or install "
            "the Tesseract system binary."
        )

    img = Image.open(BytesIO(normalized_image))

    try:
        ocr_text = pytesseract.image_to_string(img)
    except pytesseract.TesseractNotFoundError as exc:
        raise ExtractionUnavailableError(
            "Image extraction is unavailable because the Tesseract binary is not on PATH."
        ) from exc

    parsed = _extract_from_text(ocr_text)
    return ExtractedLabel(**parsed)
