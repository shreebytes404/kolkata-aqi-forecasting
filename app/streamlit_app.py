"""Small, transparent demo for the trained next-day AQI model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import METADATA_PATH, MODEL_PATH, PROCESSED_PATH  # noqa: E402

st.set_page_config(page_title="Kolkata AQI Forecast", page_icon="🌫️")
st.title("Kolkata next-day AQI forecast")
st.caption("Research demonstration only. It is not a health-alert or regulatory system.")

if not (MODEL_PATH.exists() and METADATA_PATH.exists() and PROCESSED_PATH.exists()):
    st.info("Model files are not available yet. Prepare data and train the model first; see README.md.")
    st.stop()

metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
model = joblib.load(MODEL_PATH)
data = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
stations = sorted(data["station"].dropna().unique())
station = st.selectbox("Monitoring station", stations)
station_data = data.loc[data["station"].eq(station)].sort_values("date")
latest = station_data.iloc[-1]

st.write(f"Latest model-ready observation: **{latest['date'].date()}**")
st.write("Adjust the latest available inputs below if you have newer verified readings.")

inputs: dict[str, float] = {}
for feature in metadata["features"]:
    default = float(latest[feature]) if pd.notna(latest[feature]) else float(data[feature].median())
    inputs[feature] = st.number_input(feature.replace("_", " ").title(), value=default)

if st.button("Forecast next day", type="primary"):
    prediction = float(model.predict(pd.DataFrame([inputs]))[0])
    prediction = float(np.clip(prediction, 0, 500))
    if prediction <= 50:
        category = "Good"
    elif prediction <= 100:
        category = "Satisfactory"
    elif prediction <= 200:
        category = "Moderate"
    elif prediction <= 300:
        category = "Poor"
    elif prediction <= 400:
        category = "Very Poor"
    else:
        category = "Severe"
    st.metric("Predicted AQI", f"{prediction:.0f}")
    st.success(f"Predicted category: {category}")
    st.caption(f"Selected model: {metadata['best_model']}. Validate results against the latest official station data.")
