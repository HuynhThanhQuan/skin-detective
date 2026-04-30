"""Integration tests for the inference API.

Runs against a randomly-initialised model — verifies the contract, not the medical
output. Tests are intentionally tolerant of any prediction (zero detections allowed).
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app

client = TestClient(app)


def _png_bytes(width=320, height=240, color=(200, 150, 130)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "device" in body
    assert "model_version" in body


def test_predict_returns_valid_schema():
    r = client.post("/predict", files={"file": ("img.png", _png_bytes(), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image_width"] == 320
    assert body["image_height"] == 240
    assert isinstance(body["detections"], list)
    assert body["severity_grade"] in {"mild", "moderate", "severe"}
    probs = body["severity_probabilities"]
    assert pytest.approx(sum(probs.values()), abs=1e-3) == 1.0


def test_predict_rejects_unsupported_content_type():
    r = client.post("/predict", files={"file": ("readme.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_predict_rejects_empty():
    r = client.post("/predict", files={"file": ("img.png", b"", "image/png")})
    assert r.status_code == 400
