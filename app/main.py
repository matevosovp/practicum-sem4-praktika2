from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.monitoring import FALLBACK_COUNT, RECOMMENDATION_LATENCY, REQUEST_COUNT
from app.schemas import RecommendationItem, RecommendationRequest, RecommendationResponse
from app.service import RecommendationService


MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "recommender.joblib"))
service = RecommendationService(MODEL_PATH)
app = FastAPI(title="Ecommerce Recommender API", version="1.0.0")


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        status = str(response.status_code)
    except Exception:
        REQUEST_COUNT.labels(endpoint=request.url.path, status="500").inc()
        raise
    duration = time.perf_counter() - start_time
    REQUEST_COUNT.labels(endpoint=request.url.path, status=status).inc()
    if request.url.path == "/recommendations":
        RECOMMENDATION_LATENCY.observe(duration)
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_name": service.model_name}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.post("/recommendations", response_model=RecommendationResponse)
def recommendations(payload: RecommendationRequest) -> RecommendationResponse:
    try:
        recommendations, fallback_used = service.recommend(
            user_id=payload.user_id,
            k=payload.k,
            user_history=[item.model_dump() for item in payload.user_history] if payload.user_history else None,
            exclude_item_ids=payload.exclude_item_ids,
            filter_bought=payload.filter_bought,
        )
    except Exception as exc:  # pragma: no cover - defensive boundary for API layer
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if fallback_used:
        FALLBACK_COUNT.inc()

    response_items = [RecommendationItem(**item) for item in recommendations]
    return RecommendationResponse(
        model_name=service.model_name,
        user_id=payload.user_id,
        fallback_used=fallback_used,
        recommendations=response_items,
    )


@app.exception_handler(Exception)
async def unhandled_error(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
