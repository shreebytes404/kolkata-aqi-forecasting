"""Interactive backtest demo for the trained next-day AQI model."""

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
numeric_features = [f for f in metadata["features"] if f != "station"]


def categorize(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


stations = sorted(data["station"].dropna().unique())
station = st.selectbox("Monitoring station", stations)
station_data = data.loc[data["station"].eq(station)].sort_values("date").reset_index(drop=True)

st.subheader("Backtest: see what the model would have predicted")
st.caption("Data covers 2024-01-01 to 2026-08-31. Pick any date; the forecast is for the day after it.")
dates = station_data["date"].dt.date.tolist()
selected_date = st.select_slider("Choose a date", options=dates, value=dates[-1])
row = station_data.loc[station_data["date"].dt.date.eq(selected_date)].iloc[0]

inputs = {"station": station}
for feature in numeric_features:
    inputs[feature] = float(row[feature]) if pd.notna(row[feature]) else float(data[feature].median())

prediction = float(np.clip(model.predict(pd.DataFrame([inputs]))[0], 0, 500))

col1, col2 = st.columns(2)
with col1:
    shown_aqi = f"{row['aqi']:.0f}" if pd.notna(row["aqi"]) else "N/A"
    st.metric(f"AQI on {selected_date}", shown_aqi)
with col2:
    st.metric("Model's predicted next-day AQI", f"{prediction:.0f}")
    st.caption(f"Predicted category: {categorize(prediction)}")

if pd.notna(row.get("aqi_next_day")):
    actual_next = row["aqi_next_day"]
    error = abs(prediction - actual_next)
    st.write(f"**Actual recorded AQI the next day: {actual_next:.0f}**  \nModel error: {error:.1f} AQI points")
else:
    st.write("Actual next-day AQI is not available for this date (end of the station's record).")

with st.expander("Adjust inputs manually (what-if analysis)"):
    st.caption("Try changing pollutant or weather values to see how the forecast responds.")
    manual_inputs = {"station": station}
    for feature in numeric_features:
        manual_inputs[feature] = st.number_input(
            feature.replace("_", " ").title(), value=inputs[feature], key=feature
        )
    if st.button("Recalculate forecast with manual inputs", type="primary"):
        manual_pred = float(np.clip(model.predict(pd.DataFrame([manual_inputs]))[0], 0, 500))
        st.metric("Adjusted predicted AQI", f"{manual_pred:.0f}")
        st.success(f"Predicted category: {categorize(manual_pred)}")

st.caption(
    f"Selected model: {metadata['best_model']}. This is a decision-support forecast, "
    "not a substitute for physical monitoring."
)
