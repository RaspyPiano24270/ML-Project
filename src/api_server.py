from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response
from tensorflow.keras.models import load_model  # type: ignore[reportMissingImports]

try:
    from src.pipeline_utils import load_artifact
except ModuleNotFoundError:
    from pipeline_utils import load_artifact

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent / "models"

CLASSIFIER_MODEL_PATH = MODELS_DIR / "lstm_classifier.h5"
REGRESSOR_MODEL_PATH = MODELS_DIR / "lstm_regressor.h5"
CLASSIFIER_ARTIFACT_PATH = MODELS_DIR / "lstm_classifier_artifacts.pkl"
REGRESSOR_ARTIFACT_PATH = MODELS_DIR / "lstm_regressor_artifacts.pkl"


class PredictRequest(BaseModel):
    rows: list[dict[str, float]] = Field(
        ...,
        description="Recent weather rows. Include at least `lookback` rows using model feature column names.",
    )
    classifier_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class ModelBundle:
    def __init__(self) -> None:
        self.classifier_model = None
        self.regressor_model = None
        self.classifier_artifacts: dict[str, Any] | None = None
        self.regressor_artifacts: dict[str, Any] | None = None

    def load(self) -> None:
        missing_paths = [
            path
            for path in [
                CLASSIFIER_MODEL_PATH,
                REGRESSOR_MODEL_PATH,
                CLASSIFIER_ARTIFACT_PATH,
                REGRESSOR_ARTIFACT_PATH,
            ]
            if not path.exists()
        ]
        if missing_paths:
            missing_str = ", ".join([str(path) for path in missing_paths])
            raise FileNotFoundError(f"Missing required model artifacts: {missing_str}")

        self.classifier_model = load_model(CLASSIFIER_MODEL_PATH)
        self.regressor_model = load_model(REGRESSOR_MODEL_PATH)
        self.classifier_artifacts = load_artifact(CLASSIFIER_ARTIFACT_PATH)
        self.regressor_artifacts = load_artifact(REGRESSOR_ARTIFACT_PATH)


bundle = ModelBundle()

REQUEST_COUNT = Counter(
    "stormcast_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "stormcast_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)
PREDICTION_COUNT = Counter(
    "stormcast_predictions_total",
    "Total prediction requests handled",
)

def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        bundle.load()
        yield

    app = FastAPI(title="StormCast Prediction API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        method = request.method
        path = request.url.path
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        response = await call_next(request)
        duration = time.perf_counter() - start_time
        response.headers["X-Request-ID"] = request_id

        REQUEST_COUNT.labels(method=method, path=path, status_code=str(response.status_code)).inc()
        REQUEST_LATENCY.labels(method=method, path=path).observe(duration)
        return response

    return app


app = create_app()


def _validate_and_build_sequence(rows: list[dict[str, float]], feature_columns: list[str], lookback: int, feature_scaler) -> np.ndarray:
    if len(rows) < lookback:
        raise HTTPException(
            status_code=400,
            detail=f"At least {lookback} rows are required, but received {len(rows)}",
        )

    df = pd.DataFrame(rows)
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing feature columns: {missing_cols}",
        )

    ordered = df[feature_columns].tail(lookback)
    scaled = feature_scaler.transform(ordered.values)
    return scaled.reshape(1, lookback, len(feature_columns))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    if not bundle.classifier_artifacts or not bundle.regressor_artifacts:
        raise HTTPException(status_code=500, detail="Artifacts are not loaded")

    return {
        "classifier": {
            "feature_columns": bundle.classifier_artifacts["feature_columns"],
            "lookback": bundle.classifier_artifacts["lookback"],
            "target_col": bundle.classifier_artifacts["target_col"],
        },
        "regressor": {
            "feature_columns": bundle.regressor_artifacts["feature_columns"],
            "lookback": bundle.regressor_artifacts["lookback"],
            "target_col": bundle.regressor_artifacts["target_col"],
        },
    }


@app.post("/predict/all")
def predict_all(payload: PredictRequest) -> dict[str, Any]:
    if not bundle.classifier_artifacts or not bundle.regressor_artifacts:
        raise HTTPException(status_code=500, detail="Artifacts are not loaded")

    classifier_features = bundle.classifier_artifacts["feature_columns"]
    classifier_lookback = bundle.classifier_artifacts["lookback"]
    classifier_scaler = bundle.classifier_artifacts["feature_scaler"]

    regressor_features = bundle.regressor_artifacts["feature_columns"]
    regressor_lookback = bundle.regressor_artifacts["lookback"]
    regressor_feature_scaler = bundle.regressor_artifacts["feature_scaler"]
    regressor_target_scaler = bundle.regressor_artifacts["target_scaler"]

    classifier_input = _validate_and_build_sequence(
        payload.rows,
        classifier_features,
        classifier_lookback,
        classifier_scaler,
    )
    regressor_input = _validate_and_build_sequence(
        payload.rows,
        regressor_features,
        regressor_lookback,
        regressor_feature_scaler,
    )

    classifier_prob = float(bundle.classifier_model.predict(classifier_input, verbose=0)[0][0])
    classifier_class = int(classifier_prob >= payload.classifier_threshold)

    regression_scaled = bundle.regressor_model.predict(regressor_input, verbose=0)
    regression_value = float(regressor_target_scaler.inverse_transform(regression_scaled)[0][0])
    PREDICTION_COUNT.inc()

    return {
        "classifier": {
            "probability_of_rain": classifier_prob,
            "threshold": payload.classifier_threshold,
            "predicted_class": classifier_class,
            "label": "Rain" if classifier_class == 1 else "No Rain",
        },
        "regressor": {
            "predicted_precipitation_mm": regression_value,
        },
    }
