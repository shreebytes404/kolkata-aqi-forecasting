"""Live readings from WAQI (aqicn.org). Fails soft. Token is passed in by the caller or read from WAQI_TOKEN.

IMPORTANT: WAQI reports AQI on the US EPA scale, which is NOT the Indian CPCB scale used in our recorded data.
Never mix the two. Terms: attribution to the World Air Quality Index Project and the originating EPA is mandatory;
data must not be stored/archived or redistributed, so this module caches nothing on disk.

Three ways to ask WAQI, tried in order until one returns usable readings:
  1. map/bounds  : every station inside a box around Kolkata + Howrah
  2. search      : keyword search ("Kolkata", "Howrah") with station names matched to ours
  3. feed/geo    : the nearest WAQI station to each of our station coordinates (only used if within MAX_KM)
"""
from __future__ import annotations
import math
import os
import pandas as pd, requests

BASE = "https://api.waqi.info"
BOUNDS = (22.40, 88.20, 22.80, 88.55)   # lat1, lon1, lat2, lon2 around Kolkata + Howrah
ATTRIBUTION = "Live data: World Air Quality Index Project (waqi.info) and the originating environment agencies (EPA)."
MAX_KM = 1.5
KEYWORDS = {"Ballygunge": "ballygunge", "Bidhannagar": "bidhannagar", "Fort William": "fort william",
            "Rabindra Bharati University": "rabindra bharati", "Jadavpur": "jadavpur",
            "Rabindra Sarobar": "rabindra sarobar", "Victoria": "victoria", "Belur Math": "belur",
            "Ghusuri": "ghusuri", "Padmapukur": "padmapukur"}
COORDS = {"Ballygunge": (22.5312, 88.3653), "Bidhannagar": (22.5816, 88.4168), "Fort William": (22.5539, 88.3378),
          "Rabindra Bharati University": (22.6225, 88.3789), "Jadavpur": (22.4988, 88.3713),
          "Rabindra Sarobar": (22.5121, 88.3637), "Victoria": (22.5448, 88.3426), "Belur Math": (22.6289, 88.3547),
          "Ghusuri": (22.6074, 88.3512), "Padmapukur": (22.5627, 88.3094)}


def us_category(aqi: float) -> tuple[str, str]:
    for limit, name, colour in [(50, "Good", "#2e9e4f"), (100, "Moderate", "#f2c200"),
                                (150, "Unhealthy for sensitive groups", "#f08a00"), (200, "Unhealthy", "#d9362b"),
                                (300, "Very unhealthy", "#8f3f97"), (10_000, "Hazardous", "#7e0023")]:
        if aqi <= limit:
            return name, colour
    return "Hazardous", "#7e0023"


def _km(a, b) -> float:
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    x = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(x))


def _get(path: str, params: dict, token: str):
    r = requests.get(f"{BASE}/{path}", params={**params, "token": token}, timeout=10)
    r.raise_for_status()
    return r.json()


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _match(name: str):
    n = (name or "").lower()
    return next((k for k, kw in KEYWORDS.items() if kw in n), None)


def _rows_from_list(items, info):
    """items: list of dicts from map/bounds or search."""
    rows = []
    for d in items:
        st = d.get("station") or {}
        name = st.get("name", "")
        aqi = _num(d.get("aqi"))
        ours = _match(name)
        if ours is None:
            info["unmatched"].append(name)
            continue
        if aqi is None:
            info["no_value"].append(name)
            continue
        t = d.get("time")
        obs = t.get("stime") if isinstance(t, dict) else (st.get("time") or t)
        rows.append({"station": ours, "aqi": aqi, "observed_at": obs, "waqi_name": name})
    return rows


def fetch_live(token: str | None):
    """Returns (df, info). df columns: station, aqi, observed_at, waqi_name. Fails soft."""
    info = {"status": None, "raw_count": 0, "unmatched": [], "no_value": [], "error": None, "method": None, "tried": {}}
    token = token or os.environ.get("WAQI_TOKEN")
    if not token:
        info["error"] = "No WAQI token configured."
        return pd.DataFrame(), info
    rows = []

    # 1. map/bounds
    try:
        j = _get("map/bounds/", {"latlng": ",".join(map(str, BOUNDS))}, token)
        info["status"] = j.get("status")
        if j.get("status") == "ok":
            info["raw_count"] = len(j["data"])
            rows = _rows_from_list(j["data"], info)
            info["tried"]["map/bounds"] = f"{len(j['data'])} stations, {len(rows)} usable"
        else:
            info["error"] = str(j.get("data"))[:120]
            info["tried"]["map/bounds"] = f"status {j.get('status')}"
    except Exception as e:  # noqa: BLE001
        info["tried"]["map/bounds"] = type(e).__name__
    if rows:
        info["method"] = "map/bounds"
    else:
        # 2. keyword search
        found = 0
        try:
            for kw in ("Kolkata", "Howrah"):
                j = _get("search/", {"keyword": kw}, token)
                if j.get("status") == "ok":
                    found += len(j["data"])
                    rows += _rows_from_list(j["data"], info)
            info["tried"]["search"] = f"{found} stations, {len(rows)} usable"
            if info["raw_count"] == 0:
                info["raw_count"] = found
        except Exception as e:  # noqa: BLE001
            info["tried"]["search"] = type(e).__name__
        if rows:
            info["method"] = "search"
        else:
            # 3. nearest WAQI station to each of our coordinates
            used = 0
            for ours, (lat, lon) in COORDS.items():
                try:
                    j = _get(f"feed/geo:{lat};{lon}/", {}, token)
                    if j.get("status") != "ok":
                        continue
                    d = j["data"]
                    geo = (d.get("city") or {}).get("geo") or []
                    aqi = _num(d.get("aqi"))
                    if aqi is None or len(geo) != 2 or _km((lat, lon), (float(geo[0]), float(geo[1]))) > MAX_KM:
                        continue
                    t = d.get("time") or {}
                    rows.append({"station": ours, "aqi": aqi, "observed_at": t.get("s"), "waqi_name": (d.get("city") or {}).get("name", "")})
                    used += 1
                except Exception:  # noqa: BLE001
                    continue
            info["tried"]["feed/geo"] = f"{used} of {len(COORDS)} stations had a WAQI station within {MAX_KM} km"
            if rows:
                info["method"] = "feed/geo"
    if not rows and not info["error"]:
        info["error"] = "WAQI returned no usable readings for our stations (see 'tried')."
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("observed_at").drop_duplicates("station", keep="last")
    return df, info
