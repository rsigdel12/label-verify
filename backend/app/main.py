import asyncio
import os
import pathlib
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.extraction.vision_client import initialize_local_ocr, rapidocr_available
from app.routes.batch import router as batch_router
from app.routes.verify import router as verify_router

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the ONNX sessions before Render marks the service ready. This keeps
    # model initialization out of the first reviewer's processing time.
    await asyncio.to_thread(initialize_local_ocr)
    yield


app = FastAPI(title="Label Verify API", lifespan=lifespan)


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.get("/ready")
async def readiness_check() -> dict:
    provider_configured = bool(os.getenv("LLM_API_KEY"))
    local_ocr_available = rapidocr_available()
    tesseract_available = shutil.which("tesseract") is not None
    ready = local_ocr_available or provider_configured or tesseract_available
    return {
        "status": "ready" if ready else "not_ready",
        "extraction": {
            "local_ocr_available": local_ocr_available,
            "vision_provider_configured": provider_configured,
            "tesseract_available": tesseract_available,
        },
    }


# Register API routes BEFORE static files (so they take priority)
app.include_router(verify_router)
app.include_router(batch_router)

# Serve static frontend files from the adjacent ../frontend folder
frontend_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    # Fallback: try relative path in case deployment structure differs
    fallback_dir = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "label-verify"
        / "frontend"
    )
    if fallback_dir.exists():
        app.mount("/", StaticFiles(directory=fallback_dir, html=True), name="frontend")
