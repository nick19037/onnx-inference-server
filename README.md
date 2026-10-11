# onnx-inference-server

A small, production-minded model-serving API: a scikit-learn classifier exported to **ONNX**, served with **FastAPI + ONNX Runtime**, with input validation, Prometheus metrics, tests, a multi-stage Docker build and CI.

## Run
```bash
pip install -r requirements-dev.txt
python serve.py export                # trains + writes model.onnx
uvicorn serve:app --port 8000
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"instances": [[5.1, 3.5, 1.4, 0.2]]}'
```
Or: `docker build -t onnx-inference-server . && docker run -p 8000:8000 onnx-inference-server`

## Endpoints
| Route | Purpose |
|---|---|
| `POST /predict` | Batched inference, returns label + class probabilities + latency |
| `GET /health` | Liveness/readiness probe |
| `GET /metrics` | Prometheus: request count by outcome, latency and batch-size histograms |

## Design notes
- **Model decoupled from runtime**: training deps are only needed to export; the serving image ships just `onnxruntime`.
- **Load once, warm up at startup** (FastAPI lifespan) so first-request latency isn't inflated.
- **Strict validation** (shape, empty, max batch) returns 422 before reaching the model.
- **Observability**: latency and batch-size histograms make it easy to alert on p95 and tune batching.
- **Multi-stage Docker, non-root user, healthcheck**; CI runs tests and builds the image.

## Possible extensions
Dynamic request batching, model versioning via `MODEL_PATH`, a Grafana dashboard, load tests (locust/k6), drift monitoring.
