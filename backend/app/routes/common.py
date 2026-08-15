import json

from fastapi import HTTPException, UploadFile
from pydantic import ValidationError

from app.models import ApplicationSubmission

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}


def parse_application_data(application_data: str) -> dict:
    try:
        payload = json.loads(application_data)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="application_data must be valid JSON."
        ) from exc
    return validate_application(payload)


def validate_application(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400, detail="Each application_data item must be a JSON object."
        )
    try:
        return ApplicationSubmission.model_validate(payload).model_dump()
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail="Application fields must be strings or null, with no unknown fields.",
        ) from exc


async def read_image_upload(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a PNG, JPEG, WEBP, or GIF image.",
        )

    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail="The uploaded image exceeds the 25 MB limit."
        )
    return image_bytes
