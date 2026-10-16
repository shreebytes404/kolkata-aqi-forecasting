"""Kolkata AQI: estimate AQI at any point from all monitoring stations, plus the original next-day forecast demo."""

from __future__ import annotations

import colorsys
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.spatial import bearing_deg, estimate, haversine_km  # noqa: E402
from src.live import ATTRIBUTION, fetch_live, us_category  # noqa: E402
from src.weather import elevations, wind_from_open_meteo, wind_series_open_meteo  # noqa: E402

APP_DATA = ROOT / "data" / "app"

st.set_page_config(page_title="Kolkata AQI", page_icon="🌫️", layout="wide")
st.title("Kolkata AQI: estimate for any location")
st.caption("Research demonstration only. Not a health-alert or regulatory system.")


def categorize(aqi: float) -> tuple[str, str]:
    for limit, name, colour in [(50, "Good", "#2e9e4f"), (100, "Satisfactory", "#8bc34a"), (200, "Moderate", "#f2c200"),
                                (300, "Poor", "#f08a00"), (400, "Very Poor", "#d9362b"), (10_000, "Severe", "#8b1a1a")]:
        if aqi <= limit:
            return name, colour
    return "Severe", "#8b1a1a"


def hex_rgb(h: str) -> list[int]:
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def season_of(month: int) -> str:
    if month in (11, 12, 1, 2):
        return "winter (Nov-Feb)"
    if month in (6, 7, 8, 9):
        return "monsoon (Jun-Sep)"
    return "pre-monsoon (Mar-May, Oct)"


@st.cache_data
def load_data():
    stations = pd.read_csv(APP_DATA / "stations.csv")
    hourly = pd.read_csv(APP_DATA / "hourly_aqi.csv.gz", parse_dates=["ts"])
    wide = hourly.pivot(index="ts", columns="station", values="aqi")
    cal = pd.read_csv(APP_DATA / "calibration.csv").set_index("season")
    loso = pd.read_csv(APP_DATA / "loso_summary.csv")
    return stations, wide, cal, loso


@st.cache_data
def load_forecast_tables():
    return (pd.read_csv(APP_DATA / "forecast_ranges.csv"), pd.read_csv(APP_DATA / "forecast_summary.csv"))


class _LiveFailed(Exception):
    """Raised so that failed fetches are never cached (Streamlit does not cache exceptions)."""


@st.cache_data(ttl=600, show_spinner="Fetching live readings...")
def _live_cached(token: str):
    df, info = fetch_live(token)
    if df.empty:
        raise _LiveFailed(info)
    return df, info


def live_readings():
    """Returns (df, info). info['token_source'] says where the token came from, without revealing it."""
    try:
        token, source = st.secrets["WAQI_TOKEN"], "found in Streamlit secrets"
    except KeyError:
        token, source = None, "key WAQI_TOKEN not found in Streamlit secrets"
    except Exception as e:  # noqa: BLE001
        token, source = None, f"Streamlit secrets could not be read ({type(e).__name__}); check the TOML format"
    token = str(token).strip().strip('"').strip("'") if token else None
    if not token:
        df, info = fetch_live(None)            # falls back to the WAQI_TOKEN environment variable, if any
        info["token_source"] = source if df.empty else "environment variable"
        return df, info
    try:
        df, info = _live_cached(token)
    except _LiveFailed as e:
        df, info = pd.DataFrame(), (e.args[0] if e.args else {"error": "unknown"})
    info["token_source"] = source
    info["token_length"] = len(token)
    return df, info


if not (APP_DATA / "hourly_aqi.csv.gz").exists():
    st.error("App data files are missing (data/app/). See README.md.")
    st.stop()

stations, wide, cal, loso = load_data()


@st.cache_data
def load_wind_obs():
    p = APP_DATA / "wind_obs.csv.gz"
    return pd.read_csv(p, parse_dates=["ts"]).set_index("ts") if p.exists() else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _om_wind(lat: float, lon: float, ts: pd.Timestamp):
    return wind_from_open_meteo(round(lat, 2), round(lon, 2), ts)


@st.cache_data(ttl=3600, show_spinner=False)
def _om_series(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp):
    return wind_series_open_meteo(round(lat, 2), round(lon, 2), start, end)


def get_wind_series(lat: float, lon: float, ts: pd.Timestamp):
    """Hourly wind from 12 h before to 24 h after ts: (DataFrame wd_deg/ws indexed by hour, source) or (None, None)."""
    start, end = ts - pd.Timedelta(hours=12), ts + pd.Timedelta(hours=24)
    obs = load_wind_obs()
    if len(obs) and start >= obs.index.min() and end <= obs.index.max():
        sub = obs.reindex(pd.date_range(start, end, freq="h"))[["wd_deg", "ws"]]
        if sub["wd_deg"].notna().mean() > 0.7:
            return sub, "station observations, city average"
    om = _om_series(lat, lon, start, end)
    return (om, "Open-Meteo") if om is not None else (None, None)


def wind_rgb(from_deg: float) -> list[int]:
    """Colour by compass direction the wind comes FROM: N red, E yellow-green, S cyan, W violet."""
    r, g, b = colorsys.hsv_to_rgb((from_deg % 360) / 360.0, 0.75, 0.9)
    return [int(r * 255), int(g * 255), int(b * 255)]


def get_wind(lat: float, lon: float, ts: pd.Timestamp):
    """(from-direction deg, speed m/s, source) or None. Station observations first, then Open-Meteo."""
    obs = load_wind_obs()
    if ts in obs.index and pd.notna(obs.loc[ts, "wd_deg"]):
        return float(obs.loc[ts, "wd_deg"]), float(obs.loc[ts, "ws"]), "station observations, city average"
    om = _om_wind(lat, lon, ts)
    return (om[0], om[1], "Open-Meteo") if om else None


@st.cache_data(ttl=86400, show_spinner=False)
def get_elevations(lats: tuple, lons: tuple):
    return elevations(lats, lons)

tab_est, tab_fc, tab_method = st.tabs(["Estimate anywhere", "Next-day forecast (backtest)", "Method and limitations"])

# ---------------------------------------------------------------- tab 1
with tab_est:
    left, right = st.columns([1, 2])
    with left:
        st.subheader("1. Choose a place")
        preset = st.selectbox("Quick pick", ["Type my own coordinates"] + stations["station"].tolist() + [
            "Howrah Station area (22.585, 88.342)", "Salt Lake Sector V (22.570, 88.430)",
            "Dum Dum Airport (22.654, 88.447)", "Behala Chowrasta (22.499, 88.318)",
            "Newtown (approx. 22.590, 88.470)"])
        defaults = {"Howrah Station area (22.585, 88.342)": (22.585, 88.342),
                    "Salt Lake Sector V (22.570, 88.430)": (22.570, 88.430),
                    "Dum Dum Airport (22.654, 88.447)": (22.654, 88.447),
                    "Behala Chowrasta (22.499, 88.318)": (22.499, 88.318),
                    "Newtown (approx. 22.590, 88.470)": (22.590, 88.470)}
        if preset in defaults:
            dlat, dlon = defaults[preset]
        elif preset in stations["station"].values:
            row = stations.loc[stations["station"].eq(preset)].iloc[0]
            dlat, dlon = float(row["latitude"]), float(row["longitude"])
        else:
            dlat, dlon = 22.5726, 88.3639
        use_here = st.checkbox("Use my current location")
        if use_here:
            try:
                from streamlit_geolocation import streamlit_geolocation
                loc = streamlit_geolocation()
            except Exception:  # noqa: BLE001
                loc = None
                st.caption("The location button is not available in this deployment.")
            if loc and loc.get("latitude") is not None:
                dlat, dlon = float(loc["latitude"]), float(loc["longitude"])
                acc = loc.get("accuracy")
                st.success(f"Using your location ({dlat:.4f}, {dlon:.4f})" + (f", accurate to about {acc:.0f} m" if acc else ""))
            else:
                st.caption("Click the button above, then allow location access in your browser.")
        qlat = st.number_input("Latitude", value=dlat, format="%.5f", step=0.001, key=f"lat_{preset}_{dlat:.4f}")
        qlon = st.number_input("Longitude", value=dlon, format="%.5f", step=0.001, key=f"lon_{preset}_{dlon:.4f}")
        if haversine_km(qlat, qlon, 22.5726, 88.3639) > 40:
            st.warning("This point is more than 40 km from central Kolkata, so the estimate is not meaningful.")
        elif preset.startswith("Newtown"):
            st.caption("Newtown has no monitoring station in our data; the estimate comes from stations several km away.")

        st.subheader("2. Choose the data")
        mode = st.radio("Data source", ["Recorded (CPCB, up to 30 Sep 2026)", "Live now (WAQI, US scale)"],
                        help="Live readings use the US AQI scale, which differs from India's CPCB scale. They are shown separately and never mixed.")
        live = mode.startswith("Live")
        if live:
            ts = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).floor("h")
            st.caption(f"Live readings for about {ts:%d %b %Y %H:%M} IST.")
        else:
            t_min, t_max = wide.index.min().to_pydatetime(), wide.index.max().to_pydatetime()
            day = st.date_input("Date", value=t_max.date(), min_value=t_min.date(), max_value=t_max.date())
            hour = st.slider("Hour (IST)", 0, 23, 12)
            ts = pd.Timestamp(day) + pd.Timedelta(hours=hour)

        with st.expander("Wind", expanded=True):
            st.caption("Wind is fetched for the chosen time. Our tests found that weighting stations by wind did not "
                       "improve accuracy (see Method tab), so weighting is off by default.")
            use_wind = st.checkbox("Weight stations by wind (experimental)", value=False)
            manual_wind = st.checkbox("Set wind direction manually", value=False)
            manual_dir = st.slider("Wind blowing FROM (degrees, 0=N, 90=E)", 0, 359, 180, disabled=not manual_wind)

    with right:
        cat = us_category if live else categorize
        if live:
            live_df, live_info = live_readings()
            st.caption(ATTRIBUTION)
            with st.expander("Live feed details"):
                st.write({k: v for k, v in live_info.items() if k != "unmatched"})
                st.caption("token_source tells you whether the app can see your secret. The token itself is never shown.")
                st.write("Stations found by WAQI but not matched to ours:", live_info.get("unmatched", [])[:25])
                if not live_df.empty:
                    st.dataframe(live_df, hide_index=True)
            if live_df.empty:
                st.warning("No live readings available right now" + (f" ({live_info['error']})" if live_info.get("error") else "")
                           + ". Switch to recorded data.")
                st.stop()
            readings = live_df.set_index("station")["aqi"].reindex(stations["station"])
        else:
            if ts not in wide.index:
                st.warning("No data for that time.")
                st.stop()
            readings = wide.loc[ts].reindex(stations["station"])
        s = stations.assign(aqi=readings.values)
        ok = s.dropna(subset=["aqi"])
        if len(ok) < 3:
            st.warning(f"Only {len(ok)} station(s) reported at {ts:%Y-%m-%d %H:%M}. Try another time.")
            st.stop()
        wind = get_wind(qlat, qlon, ts)
        if manual_wind:
            wind = (float(manual_dir), wind[1] if wind else float("nan"), "set manually")
        wind_dir = wind[0] if wind else None
        if use_wind and wind_dir is None:
            st.warning("No wind data for that time, so wind weighting was skipped.")
        est_plain, nearest_km, w_plain = estimate(qlat, qlon, ok["latitude"], ok["longitude"], ok["aqi"])
        est_wind, _, w_wind = estimate(qlat, qlon, ok["latitude"], ok["longitude"], ok["aqi"],
                                       wind_from_deg=wind_dir, wind_strength=1.5) if wind_dir is not None else (None, None, None)
        est, w = (est_wind, w_wind) if (use_wind and wind_dir is not None) else (est_plain, w_plain)
        season = season_of(ts.month)
        half = None if live else float(cal.loc[season, "q80"])
        lo, hi = (None, None) if live else (max(0.0, est - half), min(500.0, est + half))
        name, colour = cat(est)
        elev = get_elevations(tuple(s["latitude"]) + (qlat,), tuple(s["longitude"]) + (qlon,))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Estimated AQI", f"{est:.0f}")
        c2.metric("Likely range (80%)", "not calibrated" if live else f"{lo:.0f} to {hi:.0f}")
        c3.metric("Nearest station", f"{nearest_km:.1f} km")
        c4.metric("Altitude here", f"{elev[-1]:.0f} m" if elev else "n/a")
        st.markdown(f"<div style='padding:8px 12px;border-radius:6px;background:{colour};color:white;"
                    f"display:inline-block'>{name}</div>", unsafe_allow_html=True)
        if live:
            st.caption("Live values are on the US AQI scale (WAQI), so the CPCB-based error range does not apply.")
        else:
            st.caption(f"Range is the 80th percentile of past errors for {season}, measured by hiding each station in turn "
                       "and estimating it from the others. It is wide in winter, when AQI is volatile.")

        wser, wser_src = (None, None)
        offset = 0
        if wind_dir is not None:
            wser, wser_src = get_wind_series(qlat, qlon, ts)
            if wser is not None:
                offset = st.slider("Wind outlook: hours before (-) or after (+) the chosen time", -12, 24, 0)

        # ---- map: stations, links to the chosen point (thicker = more weight), and wind arrows
        s["colour"] = s["aqi"].map(lambda a: hex_rgb(cat(a)[1]) if pd.notna(a) else [150, 150, 150])
        s["label"] = s["aqi"].map(lambda a: f"{a:.0f}" if pd.notna(a) else "")
        links = ok.assign(qlat=qlat, qlon=qlon, weight=100 * w)
        if wind_dir is not None:
            diff = np.abs(bearing_deg(qlat, qlon, ok["latitude"].values, ok["longitude"].values) - wind_dir) % 360
            links["upwind"] = np.where(diff > 180, 360 - diff, diff) < 60
        else:
            links["upwind"] = False
        links["lcolour"] = links["upwind"].map(lambda u: [214, 69, 65, 200] if u else [90, 110, 140, 130])
        layers = [
            pdk.Layer("LineLayer", data=links, get_source_position="[longitude, latitude]",
                      get_target_position="[qlon, qlat]", get_color="lcolour", get_width="1 + weight / 8"),
            pdk.Layer("ScatterplotLayer", data=s, get_position="[longitude, latitude]", get_fill_color="colour",
                      get_radius=450, pickable=True, stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1),
            pdk.Layer("TextLayer", data=s, get_position="[longitude, latitude]", get_text="label", get_size=14,
                      get_color=[0, 0, 0], get_pixel_offset=[0, -18]),
            pdk.Layer("ScatterplotLayer", data=pd.DataFrame({"longitude": [qlon], "latitude": [qlat]}),
                      get_position="[longitude, latitude]", get_fill_color=[20, 20, 20], get_radius=300),
        ]
        shown_dir = wind_dir
        if wind_dir is not None and wser is not None:
            when = ts + pd.Timedelta(hours=offset)
            if when in wser.index and pd.notna(wser.loc[when, "wd_deg"]):
                shown_dir = float(wser.loc[when, "wd_deg"])
        if shown_dir is not None:
            toward = (shown_dir + 180) % 360  # direction the air is moving TOWARD
            gl = [(qlat + dy, qlon + dx) for dy in (-0.06, 0, 0.06) for dx in (-0.08, 0, 0.08)]
            arrows = pd.DataFrame({"latitude": [g[0] for g in gl], "longitude": [g[1] for g in gl],
                                   "angle": 90 - toward, "t": "➤", "rgb": [wind_rgb(shown_dir) + [210]] * len(gl)})
            layers.append(pdk.Layer("TextLayer", data=arrows, get_position="[longitude, latitude]", get_text="t",
                                    get_angle="angle", get_size=30, get_color="rgb"))
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=qlat, longitude=qlon, zoom=10.5),
                                 tooltip={"text": "{station}\nAQI {aqi}"}), height=460)
        if wind_dir is not None:
            spd = f"{wind[1]:.1f} m/s" if not np.isnan(wind[1]) else "speed n/a"
            st.caption(f"Arrow colour = the direction the wind comes FROM; arrows point where the air is going "
                       f"({wind[2]}: from {wind_dir:.0f}°, {spd}). Dots show each station's AQI. Lines run from each station to "
                       "your point: thicker = more weight, red = upwind of your point.")
            chips = "".join(f"<span style='display:inline-block;margin:2px 6px 2px 0;padding:2px 8px;border-radius:10px;"
                            f"background:rgb({','.join(map(str, wind_rgb(a)))});color:#111;font-size:12px'>{n} {a:.0f}°</span>"
                            for n, a in [("N", 0), ("NE", 45), ("E", 90), ("SE", 135), ("S", 180), ("SW", 225), ("W", 270), ("NW", 315)])
            st.markdown("<div>Colour key (wind coming from): " + chips + "</div>", unsafe_allow_html=True)
            if wser is not None:
                st.markdown("**How the wind changes around this time** (" + wser_src + "; the AQI estimate itself stays at the chosen hour)")
                cells = []
                for k in range(-12, 25, 3):
                    t = ts + pd.Timedelta(hours=k)
                    if t in wser.index and pd.notna(wser.loc[t, "wd_deg"]):
                        a, sp = float(wser.loc[t, "wd_deg"]), wser.loc[t, "ws"]
                        bold = "border:2px solid #111;" if k == offset else ""
                        cells.append(f"<div style='display:inline-block;text-align:center;margin:2px 8px 2px 0;{bold}padding:2px 6px;border-radius:6px'>"
                                     f"<div style='font-size:11px'>{t:%a %H}h</div>"
                                     f"<div style='font-size:26px;line-height:26px;color:rgb({','.join(map(str, wind_rgb(a)))});"
                                     f"transform:rotate({(a + 180) % 360 - 90:.0f}deg)'>➤</div>"
                                     f"<div style='font-size:11px'>{a:.0f}°, {sp:.1f} m/s</div></div>")
                st.markdown("".join(cells), unsafe_allow_html=True)
                st.caption("Wind is a single value for the whole city at each hour. It varies very little across Kolkata, "
                           "so the arrows on the map are all the same; the change you can see is over time.")
        else:
            st.caption("Dots show each station's AQI. Lines run from each station to your point (thicker = more weight). "
                       "Wind data is not available for this time.")

        st.subheader("What happens next (forecast)")
        if live:
            st.info("Forecasts are only shown for recorded CPCB data, where we measured how accurate they are.")
        else:
            fr, fsum = load_forecast_tables()
            rows = []
            for hz in (1, 3, 6, 24):
                q = float(fr[(fr["horizon_h"] == hz) & (fr["season"] == season)]["q80"].iloc[0])
                rows.append({"In": f"{hz} hour" + ("s" if hz > 1 else ""), "Forecast AQI": round(est),
                             "Likely range (80%)": f"{max(0, est - q):.0f} to {min(500, est + q):.0f}"})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("This is the 'it stays the same' forecast with a range from past changes in the same season. "
                       "We also trained a machine-learning model; it was no better overall (see Method tab), "
                       "so we show the simpler forecast.")

        st.subheader("How wind changes the estimate")
        if est_wind is not None:
            a1, a2, a3 = st.columns(3)
            a1.metric("Without wind", f"{est_plain:.0f}")
            a2.metric("With wind weighting", f"{est_wind:.0f}", delta=f"{est_wind - est_plain:+.1f}")
            a3.metric("Wind", f"{wind_dir:.0f}° / {wind[1]:.1f} m/s" if not np.isnan(wind[1]) else f"{wind_dir:.0f}°")
            st.caption("Wind pulls the estimate toward stations that are upwind of your point. In our 55,000-hour test "
                       "this did not make estimates more accurate, so treat the difference as exploratory.")
        else:
            st.info("No wind data for this time, so the wind comparison is unavailable.")

        contrib = pd.DataFrame({"Station": ok["station"].values, "AQI": ok["aqi"].values,
                                "Distance (km)": haversine_km(qlat, qlon, ok["latitude"].values, ok["longitude"].values).round(1),
                                "Weight (%)": (100 * w).round(1)})
        if elev:
            em = dict(zip(s["station"], elev[:-1]))
            contrib["Altitude (m)"] = contrib["Station"].map(em).round(0)
        st.dataframe(contrib.sort_values("Weight (%)", ascending=False), hide_index=True)
        if len(ok) < len(s):
            st.caption("Not reporting at this hour: " + ", ".join(s.loc[s["aqi"].isna(), "station"]))

        st.subheader("Altitude")
        if elev:
            st.write(f"Your point: **{elev[-1]:.0f} m** above sea level. Stations range from "
                     f"{min(elev[:-1]):.0f} m to {max(elev[:-1]):.0f} m.")
            st.caption("Kolkata is almost flat, so altitude differences are only a few metres. It is shown for "
                       "reference and is not used in the estimate; there is nothing in this data to show it matters here.")
        else:
            st.info("Altitude lookup is unavailable right now (needs internet access to the Open-Meteo elevation service).")
        if len(ok) < len(s):
            st.caption("Not reporting at this hour: " + ", ".join(s.loc[s["aqi"].isna(), "station"]))

# ---------------------------------------------------------------- tab 2 (original model, if its files are present)
with tab_fc:
    try:
        from src.config import METADATA_PATH, MODEL_PATH, PROCESSED_PATH
        import joblib

        if not (MODEL_PATH.exists() and METADATA_PATH.exists() and PROCESSED_PATH.exists()):
            raise FileNotFoundError
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        model = joblib.load(MODEL_PATH)
        data = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
        numeric_features = [f for f in metadata["features"] if f != "station"]
        st.caption("Original model: trained on 4 stations, daily data, 2024-01-01 to 2026-08-31. Pick a date; "
                   "the forecast is for the day after it. Persistence ('tomorrow = today') is hard to beat; see Method tab.")
        station = st.selectbox("Monitoring station", sorted(data["station"].dropna().unique()))
        sd = data.loc[data["station"].eq(station)].sort_values("date").reset_index(drop=True)
        dates = sd["date"].dt.date.tolist()
        sel = st.select_slider("Choose a date", options=dates, value=dates[-1])
        row = sd.loc[sd["date"].dt.date.eq(sel)].iloc[0]
        inputs = {"station": station}
        for f in numeric_features:
            inputs[f] = float(row[f]) if pd.notna(row[f]) else float(data[f].median())
        pred = float(np.clip(model.predict(pd.DataFrame([inputs]))[0], 0, 500))
        a, b = st.columns(2)
        a.metric(f"AQI on {sel}", f"{row['aqi']:.0f}" if pd.notna(row["aqi"]) else "N/A")
        b.metric("Predicted next-day AQI", f"{pred:.0f}")
        if pd.notna(row.get("aqi_next_day")):
            st.write(f"Actual next-day AQI: **{row['aqi_next_day']:.0f}** (error {abs(pred - row['aqi_next_day']):.1f})")
    except Exception:  # noqa: BLE001
        st.info("The original next-day model files are not available in this deployment.")

# ---------------------------------------------------------------- tab 3
with tab_method:
    st.subheader("How the estimate is made")
    st.markdown(
        "- Take the latest AQI reading from every monitoring station that reported at that hour.\n"
        "- Weight each station by 1 / distance² (inverse-distance weighting), so nearer stations count more.\n"
        "- The range shown comes from past errors in the same season (see below), not a made-up formula.\n"
        "- AQI itself is the CPCB-reported value; we do not recompute it.")
    st.subheader("How well it works (hide one station, estimate it from the rest)")
    st.caption("About 55,000 hourly tests over 2024-25, 10 stations. Lower is better; units are AQI points.")
    names = {"nearest": "Copy nearest station", "idw": "Inverse-distance weighting (used here)",
             "wind_0.5": "Wind-aware, mild", "wind_1.5": "Wind-aware, medium", "wind_4": "Wind-aware, strong"}
    show = loso.assign(method=loso["method"].map(names))[["method", "MAE", "RMSE"]]
    st.dataframe(show.rename(columns={"method": "Method", "MAE": "Average error"}), hide_index=True)
    st.subheader("Typical error by season")
    st.dataframe(cal.reset_index().rename(columns={"mae": "Average error", "q50": "Median error",
                                                   "q80": "80th percentile", "q90": "90th percentile"}), hide_index=True)
    st.subheader("Forecast test: machine learning vs 'it stays the same'")
    st.caption("Expanding-window test over 2025-26, about 120,000 hourly forecasts per horizon. Improvement below 0 means worse than persistence.")
    _, fsum = load_forecast_tables()
    st.dataframe(fsum.rename(columns={"h": "Hours ahead", "variant": "Model inputs", "MAE_model": "ML error",
                                      "MAE_persistence": "Persistence error", "improvement_%": "Improvement (%)"}), hide_index=True)
    st.caption("ML gained about 3-8% at 3 hours in winter and nothing meaningful elsewhere. 'light' uses only live-style inputs; "
               "'full' also uses the previous 24 hours.")
    st.subheader("Limitations")
    st.markdown(
        "- Wind weighting did **not** improve accuracy with a single city-wide wind direction, so it is off by default.\n"
        "- Error did not grow reliably with distance from the nearest station in our tests, because all tested "
        "stations sit inside the network. Estimates far outside the station area are untested.\n"
        "- Winter errors are about three times larger than monsoon errors.\n"
        "- Recorded data ends 30 Sep 2026. Live mode needs the WAQI token and uses the US AQI scale, so it is not comparable with CPCB values.\n"
        "- Station coordinates are being verified against an official source.")
