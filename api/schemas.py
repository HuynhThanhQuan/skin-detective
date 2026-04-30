from typing import List

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Pixel-coordinate bounding box in (x, y, w, h) — same convention as COCO."""

    x: float
    y: float
    w: float
    h: float


class Detection(BaseModel):
    class_id: int = Field(..., description="Index in acne_configs.ACNE_ID")
    class_short: str = Field(..., description="Short Vietnamese label (e.g. 'thuong_viem')")
    score: float = Field(..., ge=0.0, le=1.0)
    box: BoundingBox


class GradeProbabilities(BaseModel):
    """Per-grade probabilities. Keys are grade strings ('mild' / 'moderate' / 'severe')."""

    mild: float
    moderate: float
    severe: float


class PredictResponse(BaseModel):
    image_width: int
    image_height: int
    detections: List[Detection]
    severity_grade: str
    severity_probabilities: GradeProbabilities
    inference_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_version: str
