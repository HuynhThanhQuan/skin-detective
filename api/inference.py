"""Skin Detective inference pipeline — wraps the trained Faster R-CNN detector and the
LightGBM grade classifier behind a single `predict` call.

The detector weights are loaded from `SKIN_DETECTIVE_MODEL_PATH` (default: `./models/detector.pt`).
If the file does not exist, the module loads a randomly-initialised Faster R-CNN — useful for
local smoke tests of the API surface without the trained weights.
"""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import acne_configs  # noqa: E402

MODEL_VERSION = os.environ.get("SKIN_DETECTIVE_MODEL_VERSION", "fasterrcnn-r50-v1")
MODEL_PATH = Path(os.environ.get("SKIN_DETECTIVE_MODEL_PATH", "models/detector.pt"))
SCORE_THRESHOLD = float(os.environ.get("SKIN_DETECTIVE_SCORE_THRESHOLD", "0.5"))
NUM_CLASSES = len(acne_configs.ACNE_ID)


@dataclass
class Prediction:
    image_width: int
    image_height: int
    detections: list
    severity_grade: str
    severity_probabilities: dict
    inference_ms: float


class SkinDetective:
    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self._build_model()
        self.model_loaded_from_disk = self._maybe_load_weights()
        self.model.eval().to(self.device)

    def _build_model(self) -> torch.nn.Module:
        # Init detector with the right head; weights filled in by `_maybe_load_weights`.
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        return model

    def _maybe_load_weights(self) -> bool:
        if not MODEL_PATH.exists():
            return False
        state = torch.load(MODEL_PATH, map_location="cpu")
        # Support both raw state_dict and full-model pickles (legacy).
        if isinstance(state, dict) and "state_dict" in state:
            self.model.load_state_dict(state["state_dict"])
        elif isinstance(state, dict):
            self.model.load_state_dict(state)
        else:
            self.model = state  # legacy: whole model pickle
        return True

    @torch.inference_mode()
    def predict(self, image_bytes: bytes) -> Prediction:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = image.size

        tensor = TF.to_tensor(image).to(self.device)

        t0 = time.perf_counter()
        outputs = self.model([tensor])[0]
        inference_ms = (time.perf_counter() - t0) * 1000

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        detections = []
        kept_classes: List[int] = []
        for box, score, label in zip(boxes, scores, labels):
            if score < SCORE_THRESHOLD:
                continue
            x0, y0, x1, y1 = box.tolist()
            cls = int(label)
            detections.append(
                {
                    "class_id": cls,
                    "class_short": acne_configs.ID_SHORTDESC.get(cls, f"cls_{cls}"),
                    "score": float(score),
                    "box": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
                }
            )
            kept_classes.append(cls)

        grade, grade_probs = _grade_from_detections(kept_classes)

        return Prediction(
            image_width=w,
            image_height=h,
            detections=detections,
            severity_grade=grade,
            severity_probabilities=grade_probs,
            inference_ms=inference_ms,
        )


def _grade_from_detections(class_ids: List[int]) -> Tuple[str, dict]:
    """Heuristic stand-in for the LightGBM grade classifier.

    Counts inflammatory + nodular detections and produces a soft 3-class score. Replace
    with the actual LightGBM model once weights are available — the API contract stays
    the same.
    """
    inflammatory = sum(1 for c in class_ids if c == 1)
    nodular = sum(1 for c in class_ids if c == 4)
    total = len(class_ids)
    severity_score = inflammatory * 1.5 + nodular * 3.0 + (total - inflammatory - nodular) * 0.5

    raw = np.array(
        [
            max(0.0, 5.0 - severity_score),       # mild
            max(0.0, 4.0 - abs(severity_score - 6)),  # moderate
            max(0.0, severity_score - 5.0),       # severe
        ]
    )
    if raw.sum() == 0:
        raw = np.array([1.0, 0.0, 0.0])
    probs = raw / raw.sum()
    grades = ["mild", "moderate", "severe"]
    grade = grades[int(probs.argmax())]
    return grade, {"mild": float(probs[0]), "moderate": float(probs[1]), "severe": float(probs[2])}


_singleton: SkinDetective | None = None


def get_detector() -> SkinDetective:
    """Lazy module-global; the API uses this so weights load once at first request."""
    global _singleton
    if _singleton is None:
        _singleton = SkinDetective()
    return _singleton
