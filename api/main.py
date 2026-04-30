"""FastAPI entrypoint for the Skin Detective inference service.

Run locally:
    uvicorn api.main:app --reload --port 8000

OpenAPI:
    http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .inference import MODEL_VERSION, SkinDetective, get_detector
from .schemas import (
    BoundingBox,
    Detection,
    GradeProbabilities,
    HealthResponse,
    PredictResponse,
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES = 8 * 1024 * 1024  # 8 MB

app = FastAPI(
    title="Skin Detective API",
    version="1.0.0",
    description="Acne object detection + severity grading from smartphone images.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    detector: SkinDetective = get_detector()
    return HealthResponse(
        status="ok",
        model_loaded=detector.model_loaded_from_disk,
        device=str(detector.device),
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content-type: {file.content_type}")

    payload = await file.read()
    if len(payload) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"Image too large (>{MAX_BYTES} bytes)")
    if not payload:
        raise HTTPException(status_code=400, detail="Empty upload")

    detector = get_detector()
    try:
        result = detector.predict(payload)
    except Exception as e:  # noqa: BLE001 — surface inference errors as 500
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

    return PredictResponse(
        image_width=result.image_width,
        image_height=result.image_height,
        detections=[
            Detection(
                class_id=d["class_id"],
                class_short=d["class_short"],
                score=d["score"],
                box=BoundingBox(**d["box"]),
            )
            for d in result.detections
        ],
        severity_grade=result.severity_grade,
        severity_probabilities=GradeProbabilities(**result.severity_probabilities),
        inference_ms=result.inference_ms,
        model_version=MODEL_VERSION,
    )
