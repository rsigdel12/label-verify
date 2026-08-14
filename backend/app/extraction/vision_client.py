import base64
import json
import os
import re
import time
from typing import Optional

import httpx
from PIL import Image
from io import BytesIO

from app.extraction.schema import ExtractedLabel


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

    # Warning statement: find paragraph containing 'warning'
    warning = None
    for ln in lines:
        if "warning" in ln.lower():
            warning = ln
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
        "Extract the following fields from this liquor label and return valid JSON only. "
        "The JSON object must include exactly these keys with string values: "
        "brand_name, class_type, alcohol_content, net_contents, warning_statement. "
        "If a field is not visible, use null. Do not include extra keys."
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
        async with httpx.AsyncClient(timeout=10.0) as client:
            start = time.perf_counter()
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
            latency = time.perf_counter() - start
            response.raise_for_status()
            data = response.json()
    except Exception:
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
    except Exception:
        return None

    normalized = {
        "brand_name": parsed.get("brand_name"),
        "class_type": parsed.get("class_type"),
        "alcohol_content": parsed.get("alcohol_content"),
        "net_contents": parsed.get("net_contents"),
        "warning_statement": parsed.get("warning_statement"),
    }

    return ExtractedLabel(**normalized)


async def extract_label_fields(image_bytes: bytes) -> ExtractedLabel:
    """Try provider extraction first (if key present); otherwise fall back to OCR.

    The OCR fallback uses `pytesseract` and simple heuristics to extract the
    five required fields so the app can run without paid LLM access.
    """
    # 1) Try provider if configured
    label = await _call_provider(image_bytes)
    if label is not None:
        return label

    # 2) Fallback to local OCR
    try:
        import pytesseract
    except Exception as exc:
        raise RuntimeError(
            "Tesseract OCR not available. Install pytesseract and the tesseract binary."
        ) from exc

    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise RuntimeError("Invalid image data") from exc

    try:
        ocr_text = pytesseract.image_to_string(img)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract binary not found. Install Tesseract and ensure it is on PATH."
        ) from exc

    parsed = _extract_from_text(ocr_text)
    return ExtractedLabel(**parsed)
