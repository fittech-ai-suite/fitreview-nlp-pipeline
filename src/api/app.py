"""FastAPI inference endpoint for the binary fitness sentiment model."""

import os
import sys
import time
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.inference.predictor import SentimentPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app = FastAPI(
    title="FitReview Sentiment API",
    description="Binary sentiment classification for fitness app reviews.",
    version="1.0.0",
    # disable /docs and /redoc in production via env var
    docs_url="/docs" if os.getenv("ENV") != "production" else None,
    redoc_url=None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_predictor: Optional[SentimentPredictor] = None

MAX_TEXT_LENGTH = 2000
MAX_BATCH_SIZE = 32  # tighter than before to prevent abuse


@app.on_event("startup")
def load_model() -> None:
    global _predictor
    _predictor = SentimentPredictor()
    logger.info("Model loaded.")


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    # never leak internal stack traces to the client
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)

    @validator("text")
    def strip_text(cls, v: str) -> str:
        return v.strip()


class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=MAX_BATCH_SIZE)

    @validator("texts", each_item=True)
    def validate_item(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("texts must not contain empty strings")
        if len(v) > MAX_TEXT_LENGTH:
            raise ValueError(f"each text must be at most {MAX_TEXT_LENGTH} characters")
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


@app.get("/health", tags=["Meta"], include_in_schema=False)
def health():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
@limiter.limit("60/minute")
def predict(request: Request, req: PredictRequest):
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
@limiter.limit("20/minute")
def predict_batch(request: Request, req: PredictBatchRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    t0 = time.perf_counter()
    results = _predictor.predict_batch(req.texts, batch_size=32, show_progress=False)
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
