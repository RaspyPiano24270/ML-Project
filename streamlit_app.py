from __future__ import annotations

from typing import Any

import requests
import streamlit as st

st.set_page_config(page_title="StormCast Demo", page_icon="⛈", layout="wide")
st.title("StormCast Live Inference Demo")
st.caption("Rain classification + precipitation regression with API-backed inference")


def severity_label(rain_prob: float, precip_mm: float) -> str:
    if precip_mm >= 8.0 or rain_prob >= 0.85:
        return "High"
    if precip_mm >= 3.0 or rain_prob >= 0.6:
        return "Medium"
    return "Low"


def fetch_metadata(api_base_url: str) -> dict[str, Any]:
    resp = requests.get(f"{api_base_url.rstrip('/')}/metadata", timeout=20)
    resp.raise_for_status()
    return resp.json()


def build_default_row(feature_columns: list[str]) -> dict[str, float]:
    defaults = {
        "Temperature": 22.0,
        "Humidity": 65.0,
        "Wind Speed": 12.0,
        "Precipitation": 0.0,
        "Cloud Cover": 55.0,
        "Pressure": 1012.0,
        "Rain Tomorrow": 0.0,
    }
    return {col: float(defaults.get(col, 0.0)) for col in feature_columns}


def build_input_rows(feature_columns: list[str], lookback: int) -> list[dict[str, float]]:
    st.subheader("Input Features")
    st.write("Provide a representative weather row. The app repeats it to fill the required lookback window.")

    default_row = build_default_row(feature_columns)
    cols = st.columns(3)
    row: dict[str, float] = {}

    for idx, name in enumerate(feature_columns):
        with cols[idx % 3]:
            row[name] = st.number_input(name, value=float(default_row[name]), format="%.4f")

    return [row.copy() for _ in range(lookback)]


with st.sidebar:
    st.header("Connection")
    api_base_url = st.text_input("API base URL", value="http://127.0.0.1:8000")
    threshold = st.slider("Rain threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

try:
    metadata = fetch_metadata(api_base_url)
except Exception as exc:
    st.error("Could not load API metadata. Start the FastAPI server and verify the URL.")
    st.exception(exc)
    st.stop()

classifier_info = metadata["classifier"]
regressor_info = metadata["regressor"]

classifier_features = classifier_info["feature_columns"]
classifier_lookback = int(classifier_info["lookback"])

rows = build_input_rows(classifier_features, classifier_lookback)

if st.button("Run Prediction", type="primary"):
    payload = {
        "rows": rows,
        "classifier_threshold": threshold,
    }

    try:
        response = requests.post(f"{api_base_url.rstrip('/')}/predict/all", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
    except Exception as exc:
        st.error("Prediction request failed.")
        st.exception(exc)
        st.stop()

    rain_prob = float(result["classifier"]["probability_of_rain"])
    rain_label = result["classifier"]["label"]
    precip_mm = float(result["regressor"]["predicted_precipitation_mm"])
    severity = severity_label(rain_prob, precip_mm)

    m1, m2, m3 = st.columns(3)
    m1.metric("Rain Probability", f"{rain_prob:.2%}")
    m2.metric("Predicted Precipitation (mm)", f"{precip_mm:.3f}")
    m3.metric("Storm Severity", severity)

    st.subheader("Model Outputs")
    st.json(result)

st.divider()
st.write("API model metadata")
st.json(
    {
        "classifier": {
            "lookback": classifier_lookback,
            "feature_count": len(classifier_features),
            "target": classifier_info.get("target_col"),
        },
        "regressor": {
            "lookback": int(regressor_info["lookback"]),
            "feature_count": len(regressor_info["feature_columns"]),
            "target": regressor_info.get("target_col"),
        },
    }
)
