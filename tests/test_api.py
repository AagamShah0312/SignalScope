"""Tests for the FastAPI backend."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "model_loaded" in r.json()


def test_predict_valid_image(client, sample_png_bytes):
    r = client.post("/predict", files={"file": ("t.png", sample_png_bytes, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] in ("likely_ai_generated", "likely_real", "inconclusive")
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["visualization"]["overlay"].startswith("/files/")


def test_predict_invalid_mime(client):
    r = client.post("/predict", files={"file": ("t.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_predict_corrupt_image(client):
    r = client.post("/predict", files={"file": ("t.png", b"\x89PNG not really", "image/png")})
    assert r.status_code == 400


def test_predict_missing_file(client):
    r = client.post("/predict")
    assert r.status_code in (400, 422)


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
