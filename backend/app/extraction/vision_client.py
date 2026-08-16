import asyncio
import base64
import json
import logging
import os
import re
import shutil
import threading
from io import BytesIO
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError
from rapidfuzz import fuzz

from app.extraction.errors import ExtractionUnavailableError, InvalidImageError
from app.extraction.schema import ExtractedLabel

logger = logging.getLogger(__name__)
MAX_IMAGE_PIXELS = 20_000_000
OCR_IMAGE_MAX_SIDE = 1280
OCR_DETECTION_MAX_SIDE = 960
LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "models"
LOCAL_DETECTION_MODEL = LOCAL_MODEL_DIR / "PP-OCRv6_det_tiny.onnx"
LOCAL_RECOGNITION_MODEL = LOCAL_MODEL_DIR / "PP-OCRv6_rec_tiny.onnx"
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
            # Phone photos commonly rely on EXIF orientation. Applying it before
            # OCR prevents otherwise readable labels from being processed on
            # their side.
            image = ImageOps.exif_transpose(image).convert("RGB")
            # Real bottle photos contain substantially smaller text than the
            # generated fixtures. Keep enough detail for warning text while the
            # detector's separate bound below controls inference memory.
            image.thumbnail((OCR_IMAGE_MAX_SIDE, OCR_IMAGE_MAX_SIDE))
            output = BytesIO()
            image.save(output, format="PNG", optimize=True)
            return output.getvalue()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(
            "The uploaded file is not a valid PNG, JPEG, WEBP, or GIF image."
        ) from exc


def _enhance_image(image_bytes: bytes) -> bytes:
    """Create a high-contrast OCR variant for glare and low-light photos."""
    with Image.open(BytesIO(image_bytes)) as image:
        grayscale = ImageOps.grayscale(image)
        grayscale = ImageOps.autocontrast(grayscale, cutoff=1)
        grayscale = ImageEnhance.Contrast(grayscale).enhance(1.25)
        grayscale = grayscale.filter(ImageFilter.SHARPEN)
        output = BytesIO()
        grayscale.save(output, format="PNG", optimize=True)
        return output.getvalue()


BEVERAGE_KEYWORDS = (
    "ale",
    "beer",
    "bourbon",
    "brandy",
    "cabernet",
    "chardonnay",
    "cider",
    "gin",
    "ipa",
    "lager",
    "liqueur",
    "merlot",
    "porter",
    "riesling",
    "rum",
    "sauvignon",
    "stout",
    "tequila",
    "vodka",
    "whiskey",
    "whisky",
    "wine",
)

CLASS_DESCRIPTOR_WORDS = {
    "american",
    "blended",
    "bottled",
    "bourbon",
    "by",
    "cabernet",
    "chardonnay",
    "corn",
    "dry",
    "flavored",
    "from",
    "grape",
    "kentucky",
    "malt",
    "merlot",
    "pinot",
    "red",
    "riesling",
    "rose",
    "rye",
    "sauvignon",
    "sparkling",
    "straight",
    "the",
    "white",
}


def _is_utility_line(line: str) -> bool:
    return bool(
        re.search(
            r"\b(?:government\s+warning|net\s*(?:contents?)?|alc(?:ohol)?|"
            r"proof|distilled|bottled|produced|imported|batch|lot|barcode)\b",
            line,
            re.IGNORECASE,
        )
        or re.search(r"\d{1,3}(?:\.\d+)?\s*%", line)
    )


def _looks_like_class_fragment(line: str) -> bool:
    words = set(re.findall(r"[a-z]+", line.lower()))
    known_words = CLASS_DESCRIPTOR_WORDS | set(BEVERAGE_KEYWORDS)
    return bool(words) and (
        any(keyword in line.lower() for keyword in BEVERAGE_KEYWORDS)
        or words <= CLASS_DESCRIPTOR_WORDS
        # OCR mistakes in a regulated designation (for example BOURBAN)
        # should stay attached to the adjacent beverage word so comparison can
        # correctly flag them for review.
        or all(
            max(fuzz.ratio(word, known_word) for known_word in known_words) >= 80
            for word in words
        )
    )


# Heuristics for parsing OCR text into fields. Optional line heights let the
# parser use the visual hierarchy of a label instead of assuming that the very
# first detected line is always the brand.
def _extract_from_text(text: str, line_heights: list[float] | None = None) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    joined = "\n".join(lines)

    alcohol_line = next(
        (index for index, line in enumerate(lines) if re.search(r"\d(?:\.\d+)?\s*%", line)),
        None,
    )
    warning_line = next(
        (index for index, line in enumerate(lines) if "warning" in line.lower()),
        None,
    )
    brand_search_end = min(
        (index for index in (alcohol_line, warning_line) if index is not None),
        default=len(lines),
    )

    brand_index = 0 if lines else None
    if lines and line_heights and len(line_heights) == len(lines):
        candidates = [
            index
            for index, line in enumerate(lines[:brand_search_end])
            if not _is_utility_line(line)
            and "warning" not in line.lower()
            and not re.match(r"^\(?\d\)?[.)]?\s", line)
        ]
        if candidates:
            # Prefer the largest detected text, with a small top-of-label tie
            # breaker. Perspective can make the line below a brand appear a
            # few pixels taller, so choose the earliest candidate within 15%
            # of the maximum instead of requiring the absolute maximum.
            maximum_height = max(line_heights[index] for index in candidates)
            brand_index = next(
                index
                for index in candidates
                if line_heights[index] >= maximum_height * 0.85
            )
    brand = lines[brand_index] if brand_index is not None else None

    # Class/type normally sits between the brand and alcohol declaration. OCR
    # commonly returns a wrapped designation as two boxes, so join those lines.
    class_type = None
    search_end = alcohol_line if alcohol_line is not None else min(len(lines), 8)
    designation_indexes = [
        index
        for index in range(search_end)
        if index != brand_index
        and not _is_utility_line(lines[index])
        and any(keyword in lines[index].lower() for keyword in BEVERAGE_KEYWORDS)
    ]
    if designation_indexes:
        start = end = designation_indexes[0]
        while start > 0 and start - 1 != brand_index and _looks_like_class_fragment(
            lines[start - 1]
        ):
            start -= 1
        while end + 1 < search_end and end + 1 != brand_index and _looks_like_class_fragment(
            lines[end + 1]
        ):
            end += 1
        class_type = " ".join(lines[start : end + 1])
    elif search_end:
        fallback_lines = [
            line
            for index, line in enumerate(lines[:search_end])
            if index != brand_index and not _is_utility_line(line)
        ]
        if fallback_lines:
            # Class/type usually appears immediately above the alcohol
            # declaration, so the last plausible line is safer than line two.
            class_type = fallback_lines[-1]

    # Alcohol content (ABV)
    abv_match = re.search(
        r"((?:alc(?:ohol)?\.?\s*)?\d{1,3}(?:\.\d+)?\s*%\s*"
        r"(?:(?:alc(?:ohol)?\.?\s*)?(?:by\s*)?vol(?:ume)?\.?|alc\.?/?vol\.?)?"
        r"(?:\s*\(\s*\d+(?:\.\d+)?\s*proof\s*\))?)",
        joined,
        re.IGNORECASE,
    )
    alcohol_content = abv_match.group(1) if abv_match else None

    # Net contents (ml, L)
    net_match = re.search(
        r"(\d+(?:\.\d+)?\s*(?:m[lL]|[lL]|c[lL]))\b",
        joined,
        re.IGNORECASE,
    )
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
    return LOCAL_DETECTION_MODEL.is_file() and LOCAL_RECOGNITION_MODEL.is_file()


def _get_rapidocr_engine():
    global _rapidocr_engine
    if _rapidocr_engine is None:
        with _rapidocr_init_lock:
            if _rapidocr_engine is None:
                from rapidocr import RapidOCR

                _rapidocr_engine = RapidOCR(
                    params={
                        "Global.max_side_len": OCR_IMAGE_MAX_SIDE,
                        "Global.use_cls": False,
                        "Global.log_level": "warning",
                        "Det.limit_side_len": OCR_DETECTION_MAX_SIDE,
                        "Det.limit_type": "max",
                        "Det.model_path": str(LOCAL_DETECTION_MODEL),
                        "Rec.model_path": str(LOCAL_RECOGNITION_MODEL),
                        # ONNX Runtime otherwise creates a thread per detected
                        # CPU, increasing memory sharply on constrained hosts.
                        "EngineConfig.onnxruntime.intra_op_num_threads": 1,
                        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                    }
                )
    return _rapidocr_engine


def initialize_local_ocr() -> None:
    """Load local model sessions during application startup."""
    if rapidocr_available():
        _get_rapidocr_engine()


def _run_rapidocr(image_bytes: bytes) -> dict:
    """Run the shared ONNX OCR session safely outside the async event loop."""
    engine = _get_rapidocr_engine()
    with _rapidocr_inference_lock:
        result = engine(image_bytes, use_cls=False)
    texts = tuple(result.txts or ())
    boxes = tuple(result.boxes) if result.boxes is not None else ()
    heights = [
        float(max(point[1] for point in box) - min(point[1] for point in box))
        for box in boxes
    ]
    return _extract_from_text(
        "\n".join(texts),
        heights if len(heights) == len(texts) else None,
    )


def _extraction_quality(fields: dict) -> int:
    score = sum(bool(fields.get(field)) for field in fields)
    warning = fields.get("warning_statement") or ""
    if len(warning) >= 180 and "health problems" in warning.lower():
        score += 2
    return score


def _merge_extractions(primary: dict, secondary: dict) -> dict:
    merged = dict(primary)
    for field, value in secondary.items():
        if not merged.get(field) and value:
            merged[field] = value
    if len(secondary.get("warning_statement") or "") > len(
        merged.get("warning_statement") or ""
    ):
        merged["warning_statement"] = secondary["warning_statement"]
    return merged


async def extract_label_fields(image_bytes: bytes) -> ExtractedLabel:
    """Extract fields locally, with provider and Tesseract fallbacks."""
    normalized_image = _normalize_image(image_bytes)

    # Supplying a provider key is an explicit opt-in to the higher-accuracy
    # vision path, which is particularly helpful for curved bottles,
    # perspective distortion, and decorative typography. Local OCR remains the
    # self-contained default and the provider's automatic fallback.
    if os.getenv("LLM_API_KEY"):
        label = await _call_provider(normalized_image)
        if label is not None:
            return label

    local_fields = None
    if rapidocr_available():
        try:
            local_fields = await asyncio.to_thread(_run_rapidocr, normalized_image)
            # A second high-contrast pass is only charged when the first pass
            # misses a field or truncates the long government warning.
            if _extraction_quality(local_fields) < 7:
                enhanced_image = await asyncio.to_thread(
                    _enhance_image, normalized_image
                )
                enhanced_fields = await asyncio.to_thread(
                    _run_rapidocr, enhanced_image
                )
                local_fields = _merge_extractions(local_fields, enhanced_fields)
            if any(local_fields.values()):
                return ExtractedLabel(**local_fields)
        except Exception as exc:  # Keep the optional fallbacks available.
            logger.warning("RapidOCR extraction failed: %s", type(exc).__name__)

    # This also covers unusual configurations where a provider key was added
    # after the initial check above.
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
