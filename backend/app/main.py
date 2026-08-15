import os
import pathlib
import shutil

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes.batch import router as batch_router
from app.routes.verify import router as verify_router

app = FastAPI(title="Label Verify API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.get("/ready")
async def readiness_check() -> dict:
    provider_configured = bool(os.getenv("LLM_API_KEY"))
    tesseract_available = shutil.which("tesseract") is not None
    ready = provider_configured or tesseract_available
    return {
        "status": "ready" if ready else "not_ready",
        "extraction": {
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
