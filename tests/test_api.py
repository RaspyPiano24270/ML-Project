import numpy as np
from fastapi.testclient import TestClient
from sklearn.preprocessing import MinMaxScaler

from src import api_server


class DummyClassifierModel:
    def predict(self, X, verbose=0):
        return np.array([[0.7]])


class DummyRegressorModel:
    def predict(self, X, verbose=0):
        return np.array([[0.2]])


def _fit_scalers():
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(np.array([[0.0, 0.0], [1.0, 1.0]]))

    target_scaler = MinMaxScaler()
    target_scaler.fit(np.array([[0.0], [10.0]]))
    return feature_scaler, target_scaler


def _mock_bundle_load() -> None:
    feature_scaler, target_scaler = _fit_scalers()
    api_server.bundle.classifier_model = DummyClassifierModel()
    api_server.bundle.regressor_model = DummyRegressorModel()
    api_server.bundle.classifier_artifacts = {
        "feature_columns": ["Temp", "Humidity"],
        "lookback": 2,
        "target_col": "Rain Tomorrow",
        "feature_scaler": feature_scaler,
    }
    api_server.bundle.regressor_artifacts = {
        "feature_columns": ["Temp", "Humidity"],
        "lookback": 2,
        "target_col": "Precipitation",
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }


def test_health_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(api_server.bundle, "load", _mock_bundle_load)
    with TestClient(api_server.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_and_metrics_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(api_server.bundle, "load", _mock_bundle_load)
    with TestClient(api_server.app) as client:
        payload = {
            "rows": [
                {"Temp": 0.1, "Humidity": 0.2},
                {"Temp": 0.3, "Humidity": 0.4},
            ],
            "classifier_threshold": 0.5,
        }

        pred_response = client.post("/predict/all", json=payload)
        assert pred_response.status_code == 200

        body = pred_response.json()
        assert body["classifier"]["predicted_class"] == 1
        assert "predicted_precipitation_mm" in body["regressor"]

        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        assert "stormcast_http_requests_total" in metrics_response.text
