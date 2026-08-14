import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.comparison.matcher import compare_fields
from app.extraction.vision_client import extract_label_fields

router = APIRouter()


@router.post("/verify")
async def verify_label(
    file: UploadFile = File(...),
    application_data: str = Form(...),
):
    try:
        submitted = json.loads(application_data)
    except json.JSONDecodeError as exc:  # pragma: no cover - validation path
        raise HTTPException(
            status_code=400, detail="application_data must be valid JSON."
        ) from exc

    if not isinstance(submitted, dict):
        raise HTTPException(
            status_code=400, detail="application_data must be a JSON object."
        )

    image_bytes = await file.read()
    extracted = await extract_label_fields(image_bytes)
    comparison = compare_fields(extracted, submitted)

    return {
        "filename": file.filename,
        "extracted": extracted.model_dump(),
        "comparison": comparison,
    }
