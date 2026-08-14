import asyncio
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.comparison.matcher import compare_fields
from app.extraction.vision_client import extract_label_fields

router = APIRouter()


@router.post("/verify/batch")
async def verify_batch(
    files: list[UploadFile] = File(...),
    application_data: str = Form("{}"),
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    try:
        payload = json.loads(application_data)
    except json.JSONDecodeError as exc:  # pragma: no cover - validation path
        raise HTTPException(
            status_code=400, detail="application_data must be valid JSON."
        ) from exc

    if isinstance(payload, list):
        submitted_items = payload
        if len(submitted_items) != len(files):
            raise HTTPException(
                status_code=400,
                detail="When application_data is a list, it must match the number of uploaded files.",
            )
    elif isinstance(payload, dict):
        submitted_items = [payload for _ in files]
    else:
        raise HTTPException(
            status_code=400,
            detail="application_data must be a JSON object or list of objects.",
        )

    semaphore = asyncio.Semaphore(15)

    async def process_file(file: UploadFile, submitted: dict):
        async with semaphore:
            image_bytes = await file.read()
            extracted = await extract_label_fields(image_bytes)
            comparison = compare_fields(extracted, submitted)
            return {
                "filename": file.filename,
                "extracted": extracted.model_dump(),
                "comparison": comparison,
            }

    results = await asyncio.gather(
        *(
            process_file(file, submitted_items[index])
            for index, file in enumerate(files)
        )
    )

    return {
        "results": results,
        "progress": {
            "total": len(files),
            "completed": len(results),
            "status": "complete",
        },
    }
