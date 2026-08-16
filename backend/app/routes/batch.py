import asyncio
import json
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.comparison.matcher import compare_fields
from app.extraction.errors import ExtractionUnavailableError, InvalidImageError
from app.extraction.vision_client import extract_label_fields
from app.routes.common import (
    read_image_upload,
    validate_application,
    validate_extraction_mode,
)

router = APIRouter()


@router.post("/verify/batch")
async def verify_batch(
    files: list[UploadFile] = File(...),
    application_data: str = Form("{}"),
    extraction_mode: str | None = Form(default=None),
):
    batch_started_at = time.perf_counter()
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")
    mode = validate_extraction_mode(extraction_mode)

    try:
        payload = json.loads(application_data)
    except json.JSONDecodeError as exc:  # pragma: no cover - validation path
        raise HTTPException(
            status_code=400, detail="application_data must be valid JSON."
        ) from exc

    if isinstance(payload, list):
        submitted_items = [validate_application(item) for item in payload]
        if len(submitted_items) != len(files):
            raise HTTPException(
                status_code=400,
                detail="When application_data is a list, it must match the number of uploaded files.",
            )
    elif isinstance(payload, dict):
        submitted = validate_application(payload)
        submitted_items = [submitted for _ in files]
    else:
        raise HTTPException(
            status_code=400,
            detail="application_data must be a JSON object or list of objects.",
        )

    # Free vision providers throttle bursts more aggressively than local OCR.
    # A smaller Accurate-mode pool avoids creating our own HTTP 429 failures.
    semaphore = asyncio.Semaphore(2 if mode == "vision" else 5)

    async def process_file(file: UploadFile, submitted: dict):
        file_started_at = time.perf_counter()
        async with semaphore:
            try:
                image_bytes = await read_image_upload(file)
                extracted = (
                    await extract_label_fields(image_bytes, mode)
                    if mode is not None
                    else await extract_label_fields(image_bytes)
                )
            except HTTPException as exc:
                return {
                    "filename": file.filename,
                    "error": {"status": exc.status_code, "detail": exc.detail},
                    "processing_time_ms": round(
                        (time.perf_counter() - file_started_at) * 1000, 1
                    ),
                }
            except InvalidImageError as exc:
                return {
                    "filename": file.filename,
                    "error": {"status": 400, "detail": str(exc)},
                    "processing_time_ms": round(
                        (time.perf_counter() - file_started_at) * 1000, 1
                    ),
                }
            except ExtractionUnavailableError as exc:
                return {
                    "filename": file.filename,
                    "error": {"status": 503, "detail": str(exc)},
                    "processing_time_ms": round(
                        (time.perf_counter() - file_started_at) * 1000, 1
                    ),
                }
            comparison = compare_fields(extracted, submitted)
            return {
                "filename": file.filename,
                "extracted": extracted.model_dump(),
                "comparison": comparison,
                "processing_time_ms": round(
                    (time.perf_counter() - file_started_at) * 1000, 1
                ),
            }

    results = await asyncio.gather(
        *(
            process_file(file, submitted_items[index])
            for index, file in enumerate(files)
        )
    )

    completed = sum("comparison" in result for result in results)
    failed = len(results) - completed
    return {
        "results": results,
        "progress": {
            "total": len(files),
            "completed": completed,
            "failed": failed,
            "status": "complete" if failed == 0 else "complete_with_errors",
            "processing_time_ms": round(
                (time.perf_counter() - batch_started_at) * 1000, 1
            ),
        },
    }
