"""Wind-aware spatial interpolation of AQI at any (lat, lon), plus leave-one-station-out evaluation.

Wind convention: meteorological (direction the wind blows FROM, degrees clockwise from north).
A source station is "upwind" of the query point when the bearing query->station is close to the wind direction.
"""
from __future__ import annotations
import numpy as np

R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def circular_mean_deg(angles):
    a = np.radians(np.asarray(angles, float))
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan
    return (np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) + 360) % 360


def estimate(qlat, qlon, lats, lons, aqis, wind_from_deg=None, power=2.0, wind_strength=1.5):
    """Return (estimate, nearest_km, weights). wind_strength=0 -> plain IDW. Ignores NaN sources."""
    lats, lons, aqis = map(lambda x: np.asarray(x, float), (lats, lons, aqis))
    ok = ~np.isnan(aqis)
    if not ok.any():
        return np.nan, np.nan, None
    lats, lons, aqis = lats[ok], lons[ok], aqis[ok]
    d = np.maximum(haversine_km(qlat, qlon, lats, lons), 0.05)
    f = np.ones_like(d)
    if wind_from_deg is not None and not np.isnan(wind_from_deg) and wind_strength:
        diff = np.abs(bearing_deg(qlat, qlon, lats, lons) - wind_from_deg) % 360
        diff = np.where(diff > 180, 360 - diff, diff)
        f = 1.0 + wind_strength * (np.cos(np.radians(diff)) + 1) / 2
    w = f / d ** power
    w /= w.sum()
    return float((w * aqis).sum()), float(d.min()), w
