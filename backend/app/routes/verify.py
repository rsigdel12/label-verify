import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.comparison.matcher import compare_fields
from app.extraction.errors import ExtractionUnavailableError, InvalidImageError
from app.extraction.vision_client import extract_label_fields
from app.routes.common import parse_application_data, read_image_upload

router = APIRouter()


@router.post("/verify")
async def verify_label(
    file: UploadFile = File(...),
    application_data: str = Form(...),
):
    started_at = time.perf_counter()
    submitted = parse_application_data(application_data)
    image_bytes = await read_image_upload(file)
    try:
        extracted = await extract_label_fields(image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ExtractionUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    comparison = compare_fields(extracted, submitted)

    return {
        "filename": file.filename,
        "extracted": extracted.model_dump(),
        "comparison": comparison,
        "processing_time_ms": round((time.perf_counter() - started_at) * 1000, 1),
    }
