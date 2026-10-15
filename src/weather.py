"""Wind and elevation lookups. Every function fails soft (returns None) so the app never crashes offline."""
from __future__ import annotations
import pandas as pd, requests

TIMEOUT = 8
TZ = "Asia/Kolkata"


def _get(url: str, params: dict):
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:  # noqa: BLE001
        return None


def wind_from_open_meteo(lat: float, lon: float, ts: pd.Timestamp):
    """Wind (from-direction deg, speed m/s) at hour `ts` (IST) from Open-Meteo. Returns None if unavailable."""
    today = pd.Timestamp.now(tz=TZ).tz_localize(None).normalize()
    base = {"latitude": lat, "longitude": lon, "hourly": "wind_speed_10m,wind_direction_10m",
            "wind_speed_unit": "ms", "timezone": TZ}
    if ts.normalize() < today - pd.Timedelta(days=5):
        j = _get("https://archive-api.open-meteo.com/v1/archive",
                 {**base, "start_date": ts.strftime("%Y-%m-%d"), "end_date": ts.strftime("%Y-%m-%d")})
    else:
        j = _get("https://api.open-meteo.com/v1/forecast", {**base, "past_days": 7, "forecast_days": 7})
    if not j or "hourly" not in j:
        return None
    h = pd.DataFrame(j["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    row = h.loc[h["time"].eq(ts.floor("h"))]
    if row.empty or row[["wind_direction_10m", "wind_speed_10m"]].isna().any(axis=None):
        return None
    return float(row["wind_direction_10m"].iloc[0]), float(row["wind_speed_10m"].iloc[0])


def elevations(lats, lons):
    """Metres above sea level for each point (Open-Meteo elevation API). None if unavailable."""
    lats, lons = list(map(float, lats)), list(map(float, lons))
    j = _get("https://api.open-meteo.com/v1/elevation",
             {"latitude": ",".join(f"{x:.4f}" for x in lats), "longitude": ",".join(f"{x:.4f}" for x in lons)})
    if not j or "elevation" not in j or len(j["elevation"]) != len(lats):
        return None
    return [float(e) for e in j["elevation"]]
