"""Live readings from WAQI (aqicn.org). Fails soft. Token comes from Streamlit secrets or the WAQI_TOKEN env var.

IMPORTANT: WAQI reports AQI on the US EPA scale, which is NOT the Indian CPCB scale used in our recorded data.
Never mix the two. Terms: attribution to the World Air Quality Index Project and the originating EPA is mandatory;
data must not be stored/archived or redistributed, so this module caches nothing on disk.
"""
from __future__ import annotations
import os
import pandas as pd, requests

BOUNDS = (22.40, 88.20, 22.80, 88.55)   # lat1, lon1, lat2, lon2 around Kolkata + Howrah
ATTRIBUTION = "Live data: World Air Quality Index Project (waqi.info) and the originating environment agencies (EPA)."
KEYWORDS = {"Ballygunge": "ballygunge", "Bidhannagar": "bidhannagar", "Fort William": "fort william",
            "Rabindra Bharati University": "rabindra bharati", "Jadavpur": "jadavpur",
            "Rabindra Sarobar": "rabindra sarobar", "Victoria": "victoria", "Belur Math": "belur",
            "Ghusuri": "ghusuri", "Padmapukur": "padmapukur"}


def us_category(aqi: float) -> tuple[str, str]:
    for limit, name, colour in [(50, "Good", "#2e9e4f"), (100, "Moderate", "#f2c200"),
                                (150, "Unhealthy for sensitive groups", "#f08a00"), (200, "Unhealthy", "#d9362b"),
                                (300, "Very unhealthy", "#8f3f97"), (10_000, "Hazardous", "#7e0023")]:
        if aqi <= limit:
            return name, colour
    return "Hazardous", "#7e0023"


def fetch_live(token: str | None):
    """Returns (matched_df, info). matched_df columns: station, aqi, observed_at. info = dict for debugging."""
    info = {"status": None, "raw_count": 0, "unmatched": [], "error": None}
    token = token or os.environ.get("WAQI_TOKEN")
    if not token:
        info["error"] = "No WAQI token configured."
        return pd.DataFrame(), info
    try:
        r = requests.get("https://api.waqi.info/map/bounds/", params={"latlng": ",".join(map(str, BOUNDS)), "token": token}, timeout=10)
        r.raise_for_status()
        j = r.json()
    except Exception as e:  # noqa: BLE001
        info["error"] = type(e).__name__
        return pd.DataFrame(), info
    info["status"] = j.get("status")
    if j.get("status") != "ok":
        info["error"] = str(j.get("data"))[:120]
        return pd.DataFrame(), info
    rows, info["raw_count"] = [], len(j["data"])
    for d in j["data"]:
        name = (d.get("station") or {}).get("name", "")
        try:
            aqi = float(d["aqi"])
        except (KeyError, TypeError, ValueError):
            continue                                      # "-" means no current value
        ours = next((k for k, kw in KEYWORDS.items() if kw in name.lower()), None)
        if ours is None:
            info["unmatched"].append(name)
            continue
        rows.append({"station": ours, "aqi": aqi, "observed_at": (d.get("station") or {}).get("time"), "waqi_name": name})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("observed_at").drop_duplicates("station", keep="last")
    return df, info
