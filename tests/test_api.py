"""Smoke tests for the FastAPI inference endpoints."""

import os
import pytest
from fastapi.testclient import TestClient

os.environ["SKIP_MODEL_LOAD"] = "true"

from src.api.app import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_no_model_returns_503():
    resp = client.post("/predict", json={"text": "Great app, love it!"})
    assert resp.status_code == 503


def test_predict_batch_no_model_returns_503():
    resp = client.post("/predict/batch", json={"texts": ["Great app!", "Terrible app."]})
    assert resp.status_code == 503


def test_predict_empty_text_rejected():
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422


def test_predict_text_too_long_rejected():
    resp = client.post("/predict", json={"text": "x" * 2001})
    assert resp.status_code == 422


def test_predict_batch_empty_list_rejected():
    resp = client.post("/predict/batch", json={"texts": []})
    assert resp.status_code == 422


def test_predict_batch_oversized_rejected():
    resp = client.post("/predict/batch", json={"texts": ["review"] * 33})
    assert resp.status_code == 422
