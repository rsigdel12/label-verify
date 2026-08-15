import asyncio
import base64
import json
import logging
import os
import re
import shutil
import threading
from io import BytesIO
from typing import Optional

import httpx
from PIL import Image, UnidentifiedImageError

from app.extraction.errors import ExtractionUnavailableError, InvalidImageError
from app.extraction.schema import ExtractedLabel

logger = logging.getLogger(__name__)
MAX_IMAGE_PIXELS = 20_000_000
_rapidocr_engine = None
_rapidocr_init_lock = threading.Lock()
_rapidocr_inference_lock = threading.Lock()


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
            # A 650 px bound is sufficient for label text while keeping peak
            # detector memory within small container limits (for example,
            # Render's free instance).
            image.thumbnail((650, 650))
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

    # Class/type normally sits between the brand and alcohol declaration. OCR
    # commonly returns a wrapped designation as two boxes, so join those lines.
    class_type = None
    alcohol_line = next(
        (index for index, line in enumerate(lines) if re.search(r"\d(?:\.\d+)?\s*%", line)),
        None,
    )
    if alcohol_line is not None and alcohol_line > 1:
        class_type = " ".join(lines[1:alcohol_line])
    keywords = ("vodka", "whiskey", "whisky", "bourbon", "rum", "gin", "wine", "beer")
    if not class_type:
        for ln in lines[1:5]:
            if any(keyword in ln.lower() for keyword in keywords):
                class_type = ln
                break
    if not class_type and len(lines) > 1:
        class_type = lines[1]

    # Alcohol content (ABV)
    abv_match = re.search(
        r"(\d{1,3}(?:\.\d+)?\s*%\s*"
        r"(?:alc(?:ohol)?\.?\s*(?:by\s*)?vol(?:ume)?\.?|vol\.?)?"
        r"(?:\s*\(\s*\d+(?:\.\d+)?\s*proof\s*\))?)",
        joined,
        re.IGNORECASE,
    )
    alcohol_content = abv_match.group(1) if abv_match else None

    # Net contents (ml, L)
    net_match = re.search(r"(\d+(?:\.\d+)?\s*(?:m[lL]|[lL]))\b", joined)
    net_contents = net_match.group(1) if net_match else None

    # Warning statement: preserve the complete warning, even when OCR wraps it.
    warning = None
    standard_warning = re.search(
        r"(government\s+warning\s*:.*?health\s+problems\.)",
        " ".join(lines),
        flags=re.IGNORECASE,
    )
    if standard_warning:
        warning = standard_warning.group(1)
    else:
        for index, ln in enumerate(lines):
            if "warning" in ln.lower():
                warning_lines = []
                for warning_line in lines[index:]:
                    if re.match(r"^(?:batch|lot|barcode)\b", warning_line, re.IGNORECASE):
                        break
                    warning_lines.append(warning_line)
                warning = " ".join(warning_lines)
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


def rapidocr_available() -> bool:
    """Return whether the self-contained local OCR dependencies can be imported."""
    try:
        import onnxruntime  # noqa: F401
        import rapidocr  # noqa: F401
    except ImportError:
        return False
    return True


def _get_rapidocr_engine():
    global _rapidocr_engine
    if _rapidocr_engine is None:
        with _rapidocr_init_lock:
            if _rapidocr_engine is None:
                from rapidocr import RapidOCR

                _rapidocr_engine = RapidOCR(
                    params={
                        "Global.max_side_len": 650,
                        "Global.use_cls": False,
                        "Global.log_level": "warning",
                        "Det.limit_side_len": 384,
                        "Det.limit_type": "max",
                        # ONNX Runtime otherwise creates a thread per detected
                        # CPU, increasing memory sharply on constrained hosts.
                        "EngineConfig.onnxruntime.intra_op_num_threads": 1,
                        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                    }
                )
    return _rapidocr_engine


def _run_rapidocr(image_bytes: bytes) -> str:
    """Run the shared ONNX OCR session safely outside the async event loop."""
    engine = _get_rapidocr_engine()
    with _rapidocr_inference_lock:
        result = engine(image_bytes, use_cls=False)
    texts = tuple(result.txts or ())
    return "\n".join(texts)


async def extract_label_fields(image_bytes: bytes) -> ExtractedLabel:
    """Extract fields locally, with provider and Tesseract fallbacks."""
    normalized_image = _normalize_image(image_bytes)

    # The bundled ONNX path is first so deployed requests do not depend on an
    # outbound network and can stay inside the stakeholder's five-second goal.
    if rapidocr_available():
        try:
            ocr_text = await asyncio.to_thread(_run_rapidocr, normalized_image)
            if ocr_text.strip():
                return ExtractedLabel(**_extract_from_text(ocr_text))
        except Exception as exc:  # Keep the optional fallbacks available.
            logger.warning("RapidOCR extraction failed: %s", type(exc).__name__)

    # A configured vision provider can recover labels local OCR cannot read.
    label = await _call_provider(normalized_image)
    if label is not None:
        return label

    # Retain Tesseract for development environments where it is installed.
    try:
        import pytesseract
    except ImportError as exc:
        raise ExtractionUnavailableError(
            "Image extraction is unavailable. Install RapidOCR, configure LLM_API_KEY, "
            "or install Tesseract OCR."
        ) from exc

    if shutil.which("tesseract") is None:
        raise ExtractionUnavailableError(
            "Image extraction is unavailable. Install RapidOCR, configure a working "
            "LLM_API_KEY, or install the Tesseract system binary."
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
