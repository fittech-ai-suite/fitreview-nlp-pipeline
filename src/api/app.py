"""FastAPI inference endpoint for the binary fitness sentiment model."""

from __future__ import annotations

import os
import sys
import time
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.inference.predictor import SentimentPredictor

app = FastAPI(
    title="FitReview Sentiment API",
    description="Binary sentiment classification (positive / negative) for fitness app reviews.",
    version="1.0.0",
)

_predictor: SentimentPredictor | None = None


@app.on_event("startup")
def load_model() -> None:
    global _predictor
    _predictor = SentimentPredictor()


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, example="This app is fantastic!")

class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=128)

    @validator("texts", each_item=True)
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("texts must not contain empty strings")
        return v

class SentimentScore(BaseModel):
    negative: float
    positive: float

class PredictResponse(BaseModel):
    label: str
    label_id: int
    confidence: float
    scores: SentimentScore
    latency_ms: float

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
    count: int
    latency_ms: float


@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    t0 = time.perf_counter()
    result = _predictor.predict(req.text)
    latency = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        label=result["label"],
        label_id=result["label_id"],
        confidence=result["confidence"],
        scores=SentimentScore(**result["scores"]),
        latency_ms=round(latency, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(req: PredictBatchRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    t0 = time.perf_counter()
    results = _predictor.predict_batch(req.texts, batch_size=64, show_progress=False)
    latency = (time.perf_counter() - t0) * 1000
    predictions = [
        PredictResponse(
            label=r["label"],
            label_id=r["label_id"],
            confidence=r["confidence"],
            scores=SentimentScore(**r["scores"]),
            latency_ms=round(latency / len(results), 2),
        )
        for r in results
    ]
    return BatchPredictResponse(
        predictions=predictions,
        count=len(predictions),
        latency_ms=round(latency, 2),
    )
