import asyncio
import base64
import json
import logging
import os
import re
import shutil
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError
from rapidfuzz import fuzz

from app.extraction.errors import ExtractionUnavailableError, InvalidImageError
from app.extraction.class_types import classify_alcohol_type
from app.extraction.schema import ExtractedLabel

logger = logging.getLogger(__name__)
MAX_IMAGE_PIXELS = 20_000_000
OCR_IMAGE_MAX_SIDE = 1100
OCR_DETECTION_MAX_SIDE = 1024
OCR_PREPROCESS_MAX_SIDE = 1600
OCR_RETRY_MAX_SIDE = 960
LOCAL_OCR_RETRY_CUTOFF_SECONDS = 2.25
VISION_IMAGE_MAX_SIDE = 1600
AUTO_PROVIDER_TIMEOUT_SECONDS = 3.5
ACCURATE_TIMEOUT_SECONDS = 20.0
ACCURATE_MAX_ATTEMPTS = 2
ACCURATE_RETRY_DELAY_SECONDS = 0.4
LABEL_FIELDS = (
    "brand_name",
    "class_type",
    "alcohol_content",
    "net_contents",
    "warning_statement",
)
LABEL_JSON_SCHEMA = {
    "type": "object",
    "properties": {field: {"type": ["string", "null"]} for field in LABEL_FIELDS},
    "required": list(LABEL_FIELDS),
    "additionalProperties": False,
}
VISION_PROMPT = (
    "Read this alcohol beverage label as a compliance transcription task. Extract only "
    "text that is visibly supported by the image; never fill in likely or standard text "
    "from memory. brand_name is the printed brand, not a slogan. class_type is the exact "
    "printed beverage designation (for example Bourbon Whiskey, Vodka, Red Wine, Ale). "
    "alcohol_content must include the visible ABV/Alcohol by Volume and proof when shown. "
    "net_contents must include the number and unit. For warning_statement, transcribe the "
    "entire visible statement exactly, preserving capitalization, punctuation, numbering, "
    "and word order while joining visual line wraps with one space. Use null whenever a "
    "field is absent or not legible enough to transcribe confidently."
)
LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "models"
LOCAL_DETECTION_MODEL = LOCAL_MODEL_DIR / "PP-OCRv6_det_tiny.onnx"
LOCAL_RECOGNITION_MODEL = LOCAL_MODEL_DIR / "PP-OCRv6_rec_tiny.onnx"
_rapidocr_engine = None
_rapidocr_init_lock = threading.Lock()
_rapidocr_inference_lock = threading.Lock()


def configured_vision_provider() -> str | None:
    preferred = os.getenv("VISION_PROVIDER", "").strip().lower()
    providers = {
        "gemini": bool(os.getenv("GEMINI_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")),
    }
    if preferred in providers and providers[preferred]:
        return preferred
    # Preserve existing OpenAI deployments. A deployment with only the free
    # Gemini key automatically selects Gemini without another setting.
    if providers["openai"]:
        return "openai"
    if providers["gemini"]:
        return "gemini"
    return None


def vision_provider_configured() -> bool:
    return configured_vision_provider() is not None


def configured_vision_model() -> str | None:
    provider = configured_vision_provider()
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    if provider == "openai":
        return os.getenv("VISION_MODEL", os.getenv("LLM_MODEL", "gpt-5.4-mini"))
    return None


def _normalize_image(
    image_bytes: bytes,
    max_side: int = OCR_IMAGE_MAX_SIDE,
    output_format: str = "PNG",
) -> bytes:
    image = _load_rgb_image(image_bytes)
    image.thumbnail((max_side, max_side))
    output = BytesIO()
    if output_format == "JPEG":
        image.save(output, format="JPEG", quality=90, optimize=True)
    else:
        image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _load_rgb_image(image_bytes: bytes) -> Image.Image:
    """Validate an upload and return an EXIF-corrected RGB image."""
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
            return image
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(
            "The uploaded file is not a valid PNG, JPEG, WEBP, or GIF image."
        ) from exc


def _order_quad(points):
    import numpy as np

    ordered = np.zeros((4, 2), dtype="float32")
    point_sums = points.sum(axis=1)
    point_differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[point_sums.argmin()]
    ordered[2] = points[point_sums.argmax()]
    ordered[1] = points[point_differences.argmin()]
    ordered[3] = points[point_differences.argmax()]
    return ordered


def _find_label_quad(image):
    """Return one confident label boundary, preserving the full image if ambiguous."""
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    image_area = float(width * height)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blurred))
    lower = int(max(20, median * 0.66))
    upper = int(min(255, max(lower + 40, median * 1.33)))
    edges = cv2.Canny(blurred, lower, upper)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=2,
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        area_ratio = area / image_area
        # A phone photo often leaves the label at only 5-10% of the frame.
        # Shape and rectangularity checks below keep this lower area threshold
        # from accepting arbitrary small contours.
        if area_ratio < 0.05 or area_ratio > 0.98:
            continue
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approximation) != 4 or not cv2.isContourConvex(approximation):
            continue
        rectangle = cv2.minAreaRect(contour)
        rectangle_area = float(rectangle[1][0] * rectangle[1][1])
        rectangularity = area / rectangle_area if rectangle_area else 0.0
        if rectangularity < 0.72:
            continue
        quad = _order_quad(approximation.reshape(4, 2).astype("float32"))
        target_width = max(
            np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])
        )
        target_height = max(
            np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])
        )
        aspect_ratio = target_width / max(1.0, target_height)
        if min(target_width, target_height) < 120 or not 0.3 <= aspect_ratio <= 3.5:
            continue
        candidates.append((area_ratio * rectangularity, area, quad))

    if not candidates:
        return None
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    best_score, best_area, best_quad = candidates[0]
    if best_score < 0.04:
        return None
    # Multiple substantial panels usually mean complete flat artwork. Cropping
    # to only one would discard fields, so retain the full upload.
    if len(candidates) > 1 and candidates[1][1] >= best_area * 0.45:
        return None
    center = best_quad.mean(axis=0)
    return np.clip(
        center + (best_quad - center) * 1.025,
        [0, 0],
        [width - 1, height - 1],
    )


def _warp_quad(image, quad):
    import cv2
    import numpy as np

    ordered = _order_quad(quad)
    width = int(
        max(
            np.linalg.norm(ordered[1] - ordered[0]),
            np.linalg.norm(ordered[2] - ordered[3]),
        )
    )
    height = int(
        max(
            np.linalg.norm(ordered[3] - ordered[0]),
            np.linalg.norm(ordered[2] - ordered[1]),
        )
    )
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _prepare_ocr_image(image_bytes: bytes) -> bytes:
    """Crop and deskew a label while retaining detail from the original image."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return _enhance_image(_normalize_image(image_bytes))

    image = _load_rgb_image(image_bytes)
    rgb = np.asarray(image)

    # Boundary detection is cheaper on a bounded copy, but perspective
    # correction must use the original pixels. Warping the detection-sized
    # preview made small labels in phone photos permanently blurry before OCR.
    detection_rgb = rgb
    detection_scale = min(
        1.0, OCR_PREPROCESS_MAX_SIDE / max(image.width, image.height)
    )
    if detection_scale < 1.0:
        detection_rgb = cv2.resize(
            rgb,
            (
                max(1, round(image.width * detection_scale)),
                max(1, round(image.height * detection_scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
    quad = _find_label_quad(detection_rgb)
    if quad is not None:
        if detection_scale < 1.0:
            quad = quad / detection_scale
        rgb = _warp_quad(rgb, quad)

    height, width = rgb.shape[:2]
    # Upscaling does not invent detail, but it gives the detector enough pixels
    # to separate 3-6px lettering in screenshots and messaging-app exports.
    # The old 1.5x cap left the supplied 295px-wide label at only 375x880.
    scale = min(5.0, OCR_IMAGE_MAX_SIDE / max(width, height))
    if max(width, height) > OCR_IMAGE_MAX_SIDE or scale > 1.05:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        rgb = cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=interpolation,
        )

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    luminance = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(luminance)
    enhanced = cv2.cvtColor(
        cv2.merge((luminance, channel_a, channel_b)), cv2.COLOR_LAB2RGB
    )
    softened = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
    enhanced = cv2.addWeighted(enhanced, 1.12, softened, -0.12, 0)
    success, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 92],
    )
    return encoded.tobytes() if success else _normalize_image(image_bytes)


def _prepare_ocr_retry_image(image_bytes: bytes, region: str) -> bytes:
    """Create a higher-resolution grayscale view of a missing label region."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return _enhance_image(_normalize_image(image_bytes), region)

    image = _load_rgb_image(image_bytes)
    width, height = image.size
    if region == "upper":
        image = image.crop((0, 0, width, max(1, round(height * 0.58))))
    elif region == "lower":
        # Warning, net contents, and ABV commonly share the lower declaration
        # panel. Excluding the barcode/footer gives that small type ~2x more
        # detector pixels than retrying the entire lower 70%.
        image = image.crop(
            (0, round(height * 0.55), width, max(1, round(height * 0.86)))
        )

    retry_scale = min(5.0, OCR_RETRY_MAX_SIDE / max(image.size))
    if retry_scale != 1.0:
        image = image.resize(
            (
                max(1, round(image.width * retry_scale)),
                max(1, round(image.height * retry_scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (0, 0), 0.75)
    gray = cv2.addWeighted(gray, 1.18, blurred, -0.18, 0)
    success, encoded = cv2.imencode(".png", gray)
    if success:
        return encoded.tobytes()
    return _enhance_image(_normalize_image(image_bytes), region)


def _enhance_image(image_bytes: bytes, region: str = "full") -> bytes:
    """Create a high-contrast, targeted OCR variant without a large tensor."""
    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size
        if region == "upper":
            image = image.crop((0, 0, width, max(1, int(height * 0.72))))
        elif region == "lower":
            image = image.crop((0, int(height * 0.28), width, height))
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
    "highland",
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
    "single",
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


ALCOHOL_HINT = re.compile(
    r"(?:a[l1i]c(?:[o0]h[o0]l)?|abv|by\s+vo[l1](?:ume)?|"
    r"[a-z01]{0,4}\s*/\s*[a-z01/]{1,5}|vo[l1]\.?|proof)",
    re.IGNORECASE,
)
PERCENT_VALUE = re.compile(
    r"\b(\d{1,3}(?:[.,]\d+)?)\s*(?:%|percent\b|per\s*cent\b)",
    re.IGNORECASE,
)
PROOF_VALUE = re.compile(r"\b(\d{1,3}(?:[.,]\d+)?)\s*proof\b", re.IGNORECASE)
NET_CONTENTS_VALUE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:"
    r"m(?:l|1|[|!])?|rn(?:l|1)|millilit(?:er|re)s?|c(?:l|1)|centilit(?:er|re)s?|"
    r"l|lit(?:er|re)s?|fl\.?\s*oz\.?|fluid\s+ounces?|oz\.?)\b",
    re.IGNORECASE,
)
STANDARD_WARNING_TEXT = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not "
    "drink alcoholic beverages during pregnancy because of the risk of birth "
    "defects. (2) Consumption of alcoholic beverages impairs your ability to "
    "drive a car or operate machinery, and may cause health problems."
)


def _normalize_numeric_ocr(line: str) -> str:
    """Correct high-confidence letter/number confusions inside measurements."""
    substitutions = str.maketrans(
        {"O": "0", "o": "0", "I": "1", "i": "1", "S": "5", "s": "5", "B": "8"}
    )

    def normalize_token(match: re.Match) -> str:
        token = match.group(0)
        # Only touch OCR-looking numeric tokens containing a real digit. This
        # fixes 75O, 4O, and I2 without changing words or measurement units.
        return token.translate(substitutions) if re.search(r"\d", token) else token

    return re.sub(
        r"(?<![A-Za-z])[0-9OoIiSsB]+(?:[.,][0-9OoIiSsB]+)?",
        normalize_token,
        line,
    )


def _looks_like_alcohol_line(line: str) -> bool:
    return bool(
        PROOF_VALUE.search(line)
        or (PERCENT_VALUE.search(line) and ALCOHOL_HINT.search(line))
    )


def _extract_alcohol_content(lines: list[str]) -> tuple[str | None, int | None]:
    # Small print is often split into several adjacent OCR boxes. Check short
    # windows so "ALCOHOL / 40% / BY VOLUME" and "80 / PROOF" still parse.
    candidates = []
    for window_size in (1, 2, 3):
        for index in range(len(lines) - window_size + 1):
            combined = _normalize_numeric_ocr(
                " ".join(lines[index : index + window_size])
            )
            if _looks_like_alcohol_line(combined):
                leading_measurement = NET_CONTENTS_VALUE.match(combined)
                if leading_measurement:
                    combined = combined[leading_measurement.end() :].strip(" -|,;")
                candidates.append((combined, index))
    if candidates:

        def candidate_score(item: tuple[str, int]) -> tuple:
            value = item[0]
            has_volume = bool(
                re.search(r"\bby\s+vo[l1](?:ume)?\b", value, re.IGNORECASE)
            )
            has_proof = bool(PROOF_VALUE.search(value))
            return (
                has_volume,
                has_proof,
                -len(value) if has_volume or has_proof else 0,
                bool(
                    re.search(
                        r"\ba[l1i]c(?:ohol)?\b|\babv\b", value, re.IGNORECASE
                    )
                ),
                -len(value),
            )

        return max(candidates, key=candidate_score)
    return None, None


def _extract_net_contents(lines: list[str]) -> str | None:
    candidates = []
    for index, line in enumerate(lines):
        for candidate in (
            line,
            " ".join(lines[index : index + 2]),
            " ".join(lines[index : index + 3]),
        ):
            normalized = _normalize_numeric_ocr(candidate)
            if NET_CONTENTS_VALUE.search(normalized):
                candidates.append(normalized)
    if not candidates:
        return None
    # Prefer an explicitly labeled declaration over an isolated measurement.
    selected = max(
        candidates,
        key=lambda line: (
            bool(re.search(r"\bnet\s*(?:contents?|cont\.?)?\b", line, re.IGNORECASE)),
            -len(line),
        ),
    )
    match = NET_CONTENTS_VALUE.search(selected)
    if not match:
        return None
    value = match.group(0)
    # The final vertical stroke in "mL" is frequently lost at small sizes.
    return f"{value}L" if re.search(r"\d\s*m$", value, re.IGNORECASE) else value


def _warning_heading_score(line: str) -> float:
    normalized = " ".join(re.findall(r"[a-z0-9]+", line.casefold()))
    normalized = normalized.replace("0", "o").replace("1", "i")
    return float(fuzz.partial_ratio("government warning", normalized))


def _warning_transcription_score(value: str | None) -> float:
    """Score OCR completeness against the federally required warning wording."""
    if not value:
        return 0.0
    normalized = " ".join(re.findall(r"[a-z0-9]+", value.casefold()))
    expected = " ".join(re.findall(r"[a-z0-9]+", STANDARD_WARNING_TEXT.casefold()))
    return float(fuzz.ratio(expected, normalized))


def _warning_is_complete(value: str | None) -> bool:
    return bool(
        value
        and _warning_heading_score(value) >= 82
        and _warning_transcription_score(value) >= 90
    )


def _extract_warning(lines: list[str]) -> str | None:
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if _warning_heading_score(line) >= 82
        ),
        None,
    )
    heading_detected = start is not None
    if start is None:
        # Preserve the body when the small heading alone was unreadable. The
        # comparison layer will require manual review because the heading could
        # not be verified.
        start = next(
            (
                index
                for index, line in enumerate(lines)
                if fuzz.partial_ratio("surgeon general", line.casefold()) >= 88
            ),
            None,
        )
    if start is None:
        return None

    warning_lines = []
    for line in lines[start:]:
        if warning_lines and re.match(
            r"^(?:batch|lot|barcode|upc|produced|bottled|imported)\b",
            line,
            re.IGNORECASE,
        ):
            break
        warning_lines.append(line)
        if fuzz.partial_ratio("may cause health problems", line.casefold()) >= 88:
            break
    warning = " ".join(warning_lines)
    # A fuzzy body-anchor match can occasionally start inside unrelated label
    # copy on extremely small images. Do not return a long block of garbage as
    # the regulated warning unless the overall wording has meaningful support.
    return (
        warning
        if heading_detected or _warning_transcription_score(warning) >= 55
        else None
    )


def _looks_like_class_fragment(line: str) -> bool:
    words = set(re.findall(r"[a-z]+", line.lower()))
    known_words = CLASS_DESCRIPTOR_WORDS | set(BEVERAGE_KEYWORDS)
    looks_like_sentence = bool(
        re.match(r"^(?:this|that|a|an|our)\b", line.strip(), re.IGNORECASE)
        or len(words) > 7
    )
    return bool(words) and not looks_like_sentence and (
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
    alcohol_content, alcohol_line = _extract_alcohol_content(lines)
    warning_line = next(
        (
            index
            for index, line in enumerate(lines)
            if _warning_heading_score(line) >= 82
        ),
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
            and bool(re.search(r"[A-Za-z]", line))
            and "warning" not in line.lower()
            and not re.match(r"^\(?\d\)?[.)]?\s", line)
        ]
        if candidates:
            # Visual prominence is the strongest brand signal. Penalize a
            # recognized regulated designation so a large "VODKA" line does
            # not displace a slightly smaller brand directly above it.
            maximum_height = max(line_heights[index] for index in candidates)
            brand_index = max(
                candidates,
                key=lambda index: (
                    line_heights[index] / max(1.0, maximum_height)
                    - (0.4 if classify_alcohol_type(lines[index])[0] else 0.0)
                    - min(index, 12) * 0.025
                    - (0.15 if len(lines[index].split()) > 6 else 0.0)
                ),
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
        and classify_alcohol_type(lines[index])[0] is not None
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

    return {
        "brand_name": brand,
        "class_type": class_type,
        "alcohol_content": alcohol_content,
        "net_contents": _extract_net_contents(lines),
        "warning_statement": _extract_warning(lines),
    }


def _label_from_json(content: object) -> Optional[ExtractedLabel]:
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
        for field in LABEL_FIELDS
    }
    return ExtractedLabel(**normalized)


async def _call_openai_provider(
    image_bytes: bytes, timeout_seconds: float = ACCURATE_TIMEOUT_SECONDS
) -> Optional[ExtractedLabel]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv(
        "VISION_BASE_URL",
        os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    model = os.getenv("VISION_MODEL", os.getenv("LLM_MODEL", "gpt-5.4-mini"))
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{encoded_image}"

    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": VISION_PROMPT},
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": "original",
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "alcohol_label_extraction",
                "strict": True,
                "schema": LABEL_JSON_SCHEMA,
            }
        },
        "reasoning": {"effort": os.getenv("VISION_REASONING_EFFORT", "none")},
        "max_output_tokens": 600,
        "store": False,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        # httpx timeouts apply per network phase; the outer deadline bounds the
        # entire provider attempt so connect + upload + response cannot exceed
        # the latency budget together.
        async with asyncio.timeout(timeout_seconds):
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(
                    f"{base_url.rstrip('/')}/responses",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
    except (httpx.HTTPError, TimeoutError, ValueError, KeyError) as exc:
        logger.warning("Vision provider request failed: %s", type(exc).__name__)
        return None

    content = data.get("output_text")
    if not content:
        for output_item in data.get("output", []):
            if output_item.get("type") != "message":
                continue
            for content_item in output_item.get("content", []):
                if content_item.get("type") == "output_text":
                    content = content_item.get("text")
                    break
            if content:
                break
    if not content:
        return None

    return _label_from_json(content)


async def _call_gemini_provider(
    image_bytes: bytes, timeout_seconds: float = ACCURATE_TIMEOUT_SECONDS
) -> Optional[ExtractedLabel]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
    )
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": VISION_PROMPT},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": encoded_image,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": LABEL_JSON_SCHEMA,
            "thinkingConfig": {"thinkingLevel": "LOW"},
            "maxOutputTokens": 1200,
        },
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        for attempt in range(1, ACCURATE_MAX_ATTEMPTS + 1):
            try:
                async with asyncio.timeout(timeout_seconds):
                    response = await client.post(
                        f"{base_url.rstrip('/')}/models/{model}:generateContent",
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()
            except httpx.HTTPStatusError as exc:
                retryable = exc.response.status_code == 429 or (
                    500 <= exc.response.status_code < 600
                )
                logger.warning(
                    "Gemini vision attempt %d failed with HTTP %d.",
                    attempt,
                    exc.response.status_code,
                )
                if not retryable or attempt == ACCURATE_MAX_ATTEMPTS:
                    return None
            except (httpx.RequestError, TimeoutError, ValueError, KeyError) as exc:
                logger.warning(
                    "Gemini vision attempt %d failed: %s",
                    attempt,
                    type(exc).__name__,
                )
                if attempt == ACCURATE_MAX_ATTEMPTS:
                    return None
            else:
                try:
                    parts = data["candidates"][0]["content"]["parts"]
                    content = "".join(part.get("text", "") for part in parts)
                except (IndexError, KeyError, TypeError):
                    content = ""
                label = _label_from_json(content) if content else None
                if label is not None:
                    return label
                logger.warning(
                    "Gemini vision attempt %d returned no usable label.", attempt
                )
                if attempt == ACCURATE_MAX_ATTEMPTS:
                    return None

            # A short backoff recovers transient free-tier throttling and
            # service errors without making successful requests slower.
            await asyncio.sleep(ACCURATE_RETRY_DELAY_SECONDS * attempt)

    return None


async def _call_provider(
    image_bytes: bytes, timeout_seconds: float = ACCURATE_TIMEOUT_SECONDS
) -> Optional[ExtractedLabel]:
    provider = configured_vision_provider()
    if provider == "gemini":
        return await _call_gemini_provider(image_bytes, timeout_seconds)
    if provider == "openai":
        return await _call_openai_provider(image_bytes, timeout_seconds)
    return None


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
                        "Global.text_score": 0.4,
                        "Global.log_level": "warning",
                        "Det.limit_side_len": OCR_DETECTION_MAX_SIDE,
                        "Det.limit_type": "max",
                        "Det.thresh": 0.25,
                        "Det.box_thresh": 0.4,
                        "Det.max_candidates": 1000,
                        "Det.unclip_ratio": 1.7,
                        "Det.model_path": str(LOCAL_DETECTION_MODEL),
                        "Rec.model_path": str(LOCAL_RECOGNITION_MODEL),
                        # ONNX Runtime otherwise creates a thread per detected
                        # CPU, increasing memory sharply on constrained hosts.
                        "EngineConfig.onnxruntime.intra_op_num_threads": 2,
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
    score = int(bool(fields.get("brand_name")))
    score += int(classify_alcohol_type(fields.get("class_type"))[0] is not None)
    score += int(bool(fields.get("alcohol_content")))
    score += int(bool(fields.get("net_contents")))
    if _warning_is_complete(fields.get("warning_statement")):
        score += 3
    return score


def _alcohol_percentage(value: str | None) -> float | None:
    if not value:
        return None
    match = PERCENT_VALUE.search(_normalize_numeric_ocr(value))
    if not match:
        proof_match = PROOF_VALUE.search(_normalize_numeric_ocr(value))
        return float(proof_match.group(1).replace(",", ".")) / 2 if proof_match else None
    return float(match.group(1).replace(",", "."))


def _alcohol_is_plausible(value: str | None) -> bool:
    percentage = _alcohol_percentage(value)
    return percentage is not None and 0 < percentage <= 100


def _retry_regions(fields: dict, retry_warning: bool = True) -> tuple[str, ...]:
    """Choose no more than two detail views for fields missing from a pass."""
    missing_upper = (
        not fields.get("brand_name")
        or classify_alcohol_type(fields.get("class_type"))[0] is None
    )
    missing_lower = (
        not _alcohol_is_plausible(fields.get("alcohol_content"))
        or not fields.get("net_contents")
        or (
            retry_warning
            and not _warning_is_complete(fields.get("warning_statement"))
        )
    )
    if missing_upper and missing_lower:
        return ("upper", "lower")
    if missing_upper:
        return ("upper",)
    if missing_lower:
        return ("lower",)
    return ()


def _merge_extractions(
    primary: dict, secondary: dict, prefer_measurements: bool = False
) -> dict:
    merged = dict(primary)
    for field, value in secondary.items():
        if not merged.get(field) and value:
            merged[field] = value
    primary_warning = merged.get("warning_statement")
    secondary_warning = secondary.get("warning_statement")
    if (
        _warning_transcription_score(secondary_warning),
        len(secondary_warning or ""),
    ) > (
        _warning_transcription_score(primary_warning),
        len(primary_warning or ""),
    ):
        merged["warning_statement"] = secondary["warning_statement"]
    if _alcohol_is_plausible(secondary.get("alcohol_content")) and (
        prefer_measurements
        or not _alcohol_is_plausible(merged.get("alcohol_content"))
    ):
        merged["alcohol_content"] = secondary["alcohol_content"]
    if prefer_measurements and secondary.get("net_contents"):
        merged["net_contents"] = secondary["net_contents"]
    primary_type, _ = classify_alcohol_type(merged.get("class_type"))
    secondary_type, _ = classify_alcohol_type(secondary.get("class_type"))
    if secondary_type is not None and (
        primary_type is None
        or (
            primary_type == secondary_type
            and len(secondary.get("class_type") or "")
            > len(merged.get("class_type") or "")
        )
    ):
        merged["class_type"] = secondary["class_type"]
    return merged


async def extract_label_fields(
    image_bytes: bytes, mode: str | None = None
) -> ExtractedLabel:
    """Extract using the requested fast, accurate, or compatibility mode.

    ``local`` is the user-facing Fast mode and adds bounded detail passes only
    when the first OCR result is incomplete.
    ``vision`` is the user-facing Accurate mode and never adds local OCR latency.
    ``auto`` remains available for API compatibility and may combine both.
    """
    mode = (mode or os.getenv("EXTRACTION_MODE", "local")).strip().lower()
    if mode not in {"vision", "auto", "local"}:
        mode = "local"

    if mode == "vision" and not vision_provider_configured():
        raise ExtractionUnavailableError(
            "Accurate AI vision is not configured. Choose Fast read or configure "
            "GEMINI_API_KEY (free tier) or OPENAI_API_KEY on the server."
        )

    use_vision = mode in {"vision", "auto"} and vision_provider_configured()
    provider_fields = None

    if use_vision:
        vision_image = _normalize_image(
            image_bytes,
            max_side=VISION_IMAGE_MAX_SIDE,
            output_format="JPEG",
        )
        provider_timeout = (
            ACCURATE_TIMEOUT_SECONDS
            if mode == "vision"
            else AUTO_PROVIDER_TIMEOUT_SECONDS
        )
        provider_label = await _call_provider(vision_image, provider_timeout)
        if provider_label is not None:
            provider_fields = provider_label.model_dump()
            if mode == "vision" or _extraction_quality(provider_fields) >= 7:
                return provider_label
        elif mode == "vision":
            raise ExtractionUnavailableError(
                "Accurate AI vision did not return a result after its provider attempts. "
                "The free provider may be busy or rate-limited; try again or choose Fast read."
            )

    # Keep the provider path unchanged. Only local OCR receives the
    # confidence-gated crop, perspective correction, and contrast enhancement.
    local_started = time.perf_counter()
    normalized_image = await asyncio.to_thread(_prepare_ocr_image, image_bytes)
    local_fields = None
    if rapidocr_available():
        try:
            local_fields = await asyncio.to_thread(_run_rapidocr, normalized_image)
            if provider_fields is not None:
                merged_fields = _merge_extractions(provider_fields, local_fields)
                if any(merged_fields.values()):
                    return ExtractedLabel(**merged_fields)
            # On very small exports the statutory warning letters contain too
            # few source pixels for a trustworthy transcription. Retrying it
            # only burns CPU; still retry if ABV or net contents are missing.
            try:
                with Image.open(BytesIO(image_bytes)) as uploaded_image:
                    small_export = min(uploaded_image.size) < 600
            except (UnidentifiedImageError, OSError, ValueError):
                # Unit-test stubs and alternate decoders may not expose a PIL
                # header here. Default to the normal high-resolution behavior.
                small_export = False
            retry_warning = not small_export
            retry_regions = _retry_regions(local_fields, retry_warning)
            if small_export and "lower" not in retry_regions:
                # A focused declaration pass is cheap and more reliable than
                # measurements read from the full view of a tiny export.
                retry_regions += ("lower",)
            for region in retry_regions:
                # Regional views retain more source pixels for small type. A
                # strict cutoff prevents retries from pushing a slow machine
                # beyond the Fast-mode response target.
                if (
                    time.perf_counter() - local_started
                    >= LOCAL_OCR_RETRY_CUTOFF_SECONDS
                ):
                    break
                retry_image = await asyncio.to_thread(
                    _prepare_ocr_retry_image, image_bytes, region
                )
                retry_fields = await asyncio.to_thread(_run_rapidocr, retry_image)
                local_fields = _merge_extractions(
                    local_fields,
                    retry_fields,
                    prefer_measurements=region == "lower",
                )
                if _extraction_quality(local_fields) >= 7:
                    break
            if local_fields and any(local_fields.values()):
                return ExtractedLabel(**local_fields)
        except Exception as exc:  # Keep the optional fallbacks available.
            logger.warning("RapidOCR extraction failed: %s", type(exc).__name__)

    if provider_fields is not None and any(provider_fields.values()):
        return ExtractedLabel(**provider_fields)

    # Retain Tesseract for development environments where it is installed.
    try:
        import pytesseract
    except ImportError as exc:
        raise ExtractionUnavailableError(
            "Image extraction is unavailable. Configure GEMINI_API_KEY or "
            "OPENAI_API_KEY, install "
            "RapidOCR, or install Tesseract OCR."
        ) from exc

    if shutil.which("tesseract") is None:
        raise ExtractionUnavailableError(
            "Image extraction is unavailable. Configure a working GEMINI_API_KEY or "
            "OPENAI_API_KEY, install RapidOCR, or install the Tesseract system binary."
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
