import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.extraction.errors import ExtractionUnavailableError, InvalidImageError
from app.extraction.vision_client import extract_label_fields
from app.routes.common import read_image_upload, validate_extraction_mode

router = APIRouter()

SCAN_ADVISORY = (
    "This is an AI-assisted transcription, not a compliance decision. Confirm the "
    "detected text against the label image, especially the government warning."
)


@router.post("/scan")
async def scan_label(
    file: UploadFile = File(...),
    extraction_mode: str | None = Form(default=None),
):
    started_at = time.perf_counter()
    mode = validate_extraction_mode(extraction_mode)
    image_bytes = await read_image_upload(file)
    try:
        extracted = (
            await extract_label_fields(image_bytes, mode)
            if mode is not None
            else await extract_label_fields(image_bytes)
        )
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ExtractionUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "filename": file.filename,
        "extracted": extracted.model_dump(),
        "advisory": SCAN_ADVISORY,
        "processing_time_ms": round((time.perf_counter() - started_at) * 1000, 1),
    }
