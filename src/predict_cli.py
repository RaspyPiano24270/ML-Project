from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local StormCast inference from CSV rows.")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to CSV containing recent weather rows")
    parser.add_argument("--threshold", type=float, default=0.5, help="Rain probability threshold for classifier")
    parser.add_argument("--output-json", type=str, default="", help="Optional file path to save JSON output")
    return parser.parse_args()


def _build_sequence(df: pd.DataFrame, feature_columns: list[str], lookback: int, feature_scaler):
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    if len(df) < lookback:
        raise ValueError(f"Input CSV must contain at least {lookback} rows")

    ordered = df[feature_columns].tail(lookback)
    scaled = feature_scaler.transform(ordered.values)
    return scaled.reshape(1, lookback, len(feature_columns))


def main() -> None:
    args = parse_args()

    for path in [
        CLASSIFIER_MODEL_PATH,
        REGRESSOR_MODEL_PATH,
        CLASSIFIER_ARTIFACT_PATH,
        REGRESSOR_ARTIFACT_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required model artifact missing: {path}")

    df = pd.read_csv(args.input_csv)

    classifier_model = load_model(CLASSIFIER_MODEL_PATH)
    regressor_model = load_model(REGRESSOR_MODEL_PATH)
    classifier_artifacts: dict[str, Any] = load_artifact(CLASSIFIER_ARTIFACT_PATH)
    regressor_artifacts: dict[str, Any] = load_artifact(REGRESSOR_ARTIFACT_PATH)

    cls_input = _build_sequence(
        df,
        classifier_artifacts["feature_columns"],
        classifier_artifacts["lookback"],
        classifier_artifacts["feature_scaler"],
    )
    reg_input = _build_sequence(
        df,
        regressor_artifacts["feature_columns"],
        regressor_artifacts["lookback"],
        regressor_artifacts["feature_scaler"],
    )

    cls_prob = float(classifier_model.predict(cls_input, verbose=0)[0][0])
    cls_class = int(cls_prob >= args.threshold)

    reg_scaled = regressor_model.predict(reg_input, verbose=0)
    reg_value = float(regressor_artifacts["target_scaler"].inverse_transform(reg_scaled)[0][0])

    result = {
        "classifier": {
            "probability_of_rain": cls_prob,
            "threshold": args.threshold,
            "predicted_class": cls_class,
            "label": "Rain" if cls_class == 1 else "No Rain",
        },
        "regressor": {
            "predicted_precipitation_mm": reg_value,
        },
    }

    print(json.dumps(result, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
