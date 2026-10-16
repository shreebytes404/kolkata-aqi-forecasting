"""Multi-horizon AQI forecasts per station from hourly data.

Two model variants:
  light : only what a live reading provides (AQI now, hour, season, station, other stations' AQI now)
  full  : light + AQI lags (1,2,3,6,12,24 h) -- needs recorded history, so only usable when replaying data
Target: AQI at t+h. Baseline: persistence (AQI at t).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HORIZONS = (1, 3, 6, 24)
LAGS = (1, 2, 3, 6, 12, 24)


def make_frame(wide: pd.DataFrame, h: int | None, with_lags: bool) -> pd.DataFrame:
    """Long table: one row per (station, time). `wide` = hourly AQI, index ts, columns stations."""
    wide = wide.asfreq("h")
    stations = list(wide.columns)
    others_mean = {s: wide.drop(columns=s).mean(axis=1) for s in stations}
    parts = []
    for i, s in enumerate(stations):
        d = pd.DataFrame({"ts": wide.index, "station": s, "station_code": i, "aqi_now": wide[s].values})
        d["others_now"] = others_mean[s].values
        hr = wide.index.hour.values; doy = wide.index.dayofyear.values
        d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
        d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        if with_lags:
            for L in LAGS:
                d[f"lag_{L}"] = wide[s].shift(L).values
        if h is not None:
            d["target"] = wide[s].shift(-h).values
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def feature_cols(with_lags: bool) -> list[str]:
    base = ["station_code", "aqi_now", "others_now", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    return base + ([f"lag_{L}" for L in LAGS] if with_lags else [])


def new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, max_leaf_nodes=12, min_samples_leaf=200,
                                         l2_regularization=5.0, categorical_features=[0], random_state=42)


def fit(wide: pd.DataFrame, h: int, with_lags: bool, until: pd.Timestamp | None = None):
    df = make_frame(wide, h, with_lags).dropna(subset=["aqi_now", "target"])
    if until is not None:
        df = df[df["ts"] + pd.Timedelta(hours=h) <= until]     # no training target may reach into the test period
    cols = feature_cols(with_lags)
    # predict the CHANGE from now, so the model starts from persistence and only has to learn the correction
    return new_model().fit(df[cols], df["target"] - df["aqi_now"])


def predict_row(model, wide_row: dict, with_lags: bool) -> float:
    cols = feature_cols(with_lags)
    return float(np.clip(wide_row["aqi_now"] + model.predict(pd.DataFrame([wide_row])[cols])[0], 0, 500))
