import pytest
from fastapi.testclient import TestClient

import serve


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("m") / "model.onnx")
    serve.MODEL_PATH = path
    serve.export_model(path)
    with TestClient(serve.app) as c:  # triggers lifespan (model load)
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_known_samples(client):
    r = client.post("/predict", json={"instances": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]})
    assert r.status_code == 200
    assert [p["label"] for p in r.json()["predictions"]] == ["setosa", "virginica"]


@pytest.mark.parametrize("body", [{"instances": []}, {"instances": [[1, 2, 3]]}, {}])
def test_bad_input_rejected(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_metrics_exposed(client):
    client.post("/predict", json={"instances": [[5.1, 3.5, 1.4, 0.2]]})
    assert "inference_requests_total" in client.get("/metrics").text
