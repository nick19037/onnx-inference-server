"""Minimal ONNX model-serving API: FastAPI + ONNX Runtime + Prometheus metrics.

    python serve.py export          # train a small model and export model.onnx
    uvicorn serve:app --port 8000   # serve it
"""
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator

MODEL_PATH = os.getenv("MODEL_PATH", "model.onnx")
N_FEATURES = 4
CLASSES = ["setosa", "versicolor", "virginica"]
MAX_BATCH = int(os.getenv("MAX_BATCH", "256"))

REQUESTS = Counter("inference_requests_total", "Requests by outcome", ["status"])
LATENCY = Histogram("inference_latency_seconds", "Model inference latency")
BATCH_SIZE = Histogram("inference_batch_size", "Rows per request",
                       buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256))


def export_model(path: str = MODEL_PATH) -> None:
    """Train a baseline classifier and export it to ONNX (training deps only needed here)."""
    from skl2onnx import to_onnx
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    onnx_model = to_onnx(clf, X[:1].astype(np.float32), options={"zipmap": False})
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"exported {path} (train acc {clf.score(X, y):.3f})")


class PredictRequest(BaseModel):
    instances: List[List[float]] = Field(..., description="Rows of 4 iris measurements (cm)")

    @field_validator("instances")
    @classmethod
    def check_shape(cls, rows):
        if not rows:
            raise ValueError("instances must not be empty")
        if len(rows) > MAX_BATCH:
            raise ValueError(f"batch too large (max {MAX_BATCH})")
        if any(len(r) != N_FEATURES for r in rows):
            raise ValueError(f"each instance needs exactly {N_FEATURES} features")
        return rows


class Prediction(BaseModel):
    label: str
    probabilities: dict


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    import onnxruntime as ort  # imported lazily so `export` doesn't need it

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"{MODEL_PATH} not found - run `python serve.py export` first")
    app.state.session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    app.state.input_name = app.state.session.get_inputs()[0].name
    # warm-up so the first real request doesn't pay initialisation cost
    app.state.session.run(None, {app.state.input_name: np.zeros((1, N_FEATURES), np.float32)})
    yield


app = FastAPI(title="ONNX inference server", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x = np.asarray(req.instances, dtype=np.float32)
    BATCH_SIZE.observe(len(x))
    start = time.perf_counter()
    try:
        labels, probs = app.state.session.run(None, {app.state.input_name: x})
    except Exception as exc:  # surface model errors as 500s, but count them
        REQUESTS.labels("error").inc()
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}")
    elapsed = time.perf_counter() - start
    LATENCY.observe(elapsed)
    REQUESTS.labels("ok").inc()
    preds = [
        Prediction(label=CLASSES[int(l)], probabilities={c: round(float(p), 4) for c, p in zip(CLASSES, row)})
        for l, row in zip(labels, probs)
    ]
    return PredictResponse(predictions=preds, latency_ms=round(elapsed * 1000, 3))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__" and sys.argv[1:] == ["export"]:
    export_model()
