"""FastAPI app serving heart disease predictions.

Run locally:   uvicorn app.main:app --reload
Swagger UI:    http://localhost:8000/docs
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (FEATURE_ORDER, HealthResponse, HeartFeatures,
                         InfoResponse, PredictionResponse)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model/heart_model.joblib"))
METADATA_PATH = MODEL_PATH.parent / "metadata.json"

logger = logging.getLogger("uvicorn.error")
ml = {}   # holds the loaded model and its metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup instead of on every request."""
    try:
        ml["model"] = joblib.load(MODEL_PATH)
        ml["metadata"] = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
        logger.info("Model loaded from %s", MODEL_PATH)
    except FileNotFoundError:
        logger.error("Model not found at %s. Run `python train.py` first.", MODEL_PATH)
    yield
    ml.clear()


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts the presence of heart disease from 13 clinical features.",
    version="1.0.0",
    lifespan=lifespan,
)


def _require_model():
    if "model" not in ml:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ml["model"]


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Heart Disease Prediction API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness check used by Docker and Render."""
    loaded = "model" in ml
    return HealthResponse(status="ok" if loaded else "model_not_loaded", model_loaded=loaded)


@app.get("/info", response_model=InfoResponse)
def info():
    """Model type, expected features and training metrics."""
    model = _require_model()
    meta = ml["metadata"]
    return InfoResponse(
        model_type=type(model).__name__,
        features=meta.get("features", FEATURE_ORDER),
        target=meta.get("target_description", "1 = heart disease present, 0 = absent"),
        metrics=meta.get("metrics", {}),
        sklearn_version=meta.get("sklearn_version"),
        trained_at=meta.get("trained_at"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: HeartFeatures):
    """Predict heart disease for one patient.

    Defined with `def` (not `async def`) because model inference is CPU-bound;
    FastAPI runs it in a thread pool so it doesn't block other requests.
    """
    model = _require_model()
    features = ml["metadata"].get("features", FEATURE_ORDER)
    X = pd.DataFrame([patient.model_dump()])[features]   # same column order as training
    positive_index = list(model.classes_).index(1)
    probability = float(model.predict_proba(X)[0, positive_index])
    return PredictionResponse(heart_disease=probability >= 0.5, probability=round(probability, 4))
