"""Validated, leakage-safe construction of a daily forecasting dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import (
    DEFAULT_STATION,
    KOLKATA_LATITUDE,
    KOLKATA_LONGITUDE,
    LAGS,
    ROLLING_WINDOWS,
)

REQUIRED_AIR_COLUMNS = {"date", "station", "aqi"}
WEATHER_COLUMNS = ["temperature_2m_mean", "relative_humidity_2m_mean", "wind_speed_10m_max", "pressure_msl_mean"]


def load_air_quality(path: Path, station: str = DEFAULT_STATION) -> pd.DataFrame:
    """Load one documented daily AQI dataset and reject ambiguous input early."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Download/export daily station data first; see README.md."
        )
    frame = pd.read_csv(path)
    frame.columns = [column.strip().lower() for column in frame.columns]
    missing = REQUIRED_AIR_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["aqi"] = pd.to_numeric(frame["aqi"], errors="coerce")
    frame["station"] = frame["station"].astype(str).str.strip()
    frame = frame.loc[frame["station"].eq(station), ["date", "station", "aqi"]]
    frame = frame.dropna().sort_values("date").drop_duplicates("date", keep="last")
    frame = frame.loc[frame["aqi"].between(0, 500)].copy()
    if len(frame) < 120:
        raise ValueError("At least 120 valid daily observations are required for the first model.")
    return frame


def fetch_daily_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch reproducible historical daily weather from Open-Meteo Archive."""
    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": KOLKATA_LATITUDE,
            "longitude": KOLKATA_LONGITUDE,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "daily": ",".join(WEATHER_COLUMNS),
            "timezone": "Asia/Kolkata",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if "daily" not in payload:
        raise RuntimeError(f"Open-Meteo returned no daily weather: {json.dumps(payload)[:300]}")
    weather = pd.DataFrame(payload["daily"]).rename(columns={"time": "date"})
    weather["date"] = pd.to_datetime(weather["date"])
    return weather


def build_features(air: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Build features using strictly past AQI values; never use the future target."""
    data = air.merge(weather, on="date", how="left").sort_values("date").reset_index(drop=True)
    # Weather gaps are retained until the training pipeline imputes them within each CV fold.
    for lag in LAGS:
        data[f"aqi_lag_{lag}"] = data["aqi"].shift(lag)
    for window in ROLLING_WINDOWS:
        data[f"aqi_rolling_mean_{window}"] = data["aqi"].shift(1).rolling(window).mean()

    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["day_of_year_sin"] = np.sin(2 * np.pi * data["date"].dt.dayofyear / 365.25)
    data["day_of_year_cos"] = np.cos(2 * np.pi * data["date"].dt.dayofyear / 365.25)
    data["aqi_next_day"] = data["aqi"].shift(-1)
    return data.dropna(subset=["aqi_next_day", "aqi_lag_3", "aqi_rolling_mean_7"]).copy()
