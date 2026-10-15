"""Merge teammate-collected files into one clean hourly table per station.

Inputs  (folder passed as argv[1]): monthly hourly-AQI xlsx files, 15-min CSVs (kol-*-wbpcb-2024-25.csv), stations.csv
Output  (folder argv[2]): hourly_all_stations.csv, quality_report.csv
Wind direction is averaged as a vector (circular mean), never arithmetically.
"""
from __future__ import annotations
import glob, os, re, sys
import numpy as np, pandas as pd

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# ---- 1. hourly AQI from monthly wide xlsx (one row per day, 24 hour columns)
rows = []
for f in sorted(glob.glob(f"{SRC}/AQI_Data_Station_Level_*.xlsx")):
    sid, mon, yr = re.search(r"Site_(\d+)_.+_WBPCB_([A-Za-z]+)_(\d{4})_Hourly", os.path.basename(f)).groups()
    df = pd.read_excel(f)
    day = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    vals = df.iloc[:, 1:25].apply(pd.to_numeric, errors="coerce")
    vals.columns = range(24)
    vals["day"] = day
    long = vals.dropna(subset=["day"]).melt(id_vars="day", var_name="hour", value_name="aqi")
    base = pd.Timestamp(f"{mon} 1, {yr}")
    long["ts"] = base + pd.to_timedelta(long["day"].astype(int) - 1, unit="D") + pd.to_timedelta(long["hour"], unit="h")
    long = long[long["ts"].dt.month == base.month]          # drop padding days
    long["site_id"] = int(sid)
    rows.append(long[["site_id", "ts", "aqi"]])
aqi = pd.concat(rows).drop_duplicates(["site_id", "ts"]).sort_values(["site_id", "ts"])

# ---- 2. 15-min station CSVs -> hourly pollutants + met
RENAME = {"PM2.5 (µg/m³)": "pm25", "PM10 (µg/m³)": "pm10", "NO2 (µg/m³)": "no2", "SO2 (µg/m³)": "so2",
          "CO (mg/m³)": "co", "Ozone (µg/m³)": "o3", "NH3 (µg/m³)": "nh3", "AT (°C)": "temp_c",
          "RH (%)": "rh", "WS (m/s)": "ws", "BP (mmHg)": "bp"}
parts = []
for f in sorted(glob.glob(f"{SRC}/kol-*-wbpcb-2024-25.csv")):
    d = pd.read_csv(f, low_memory=False)
    d = d[d["Station ID"].astype(str).str.startswith("site_")]       # drop the stray '2025' row
    d["site_id"] = d["Station ID"].str.replace("site_", "").astype(int)
    # timestamps are labelled +0000 but are local IST wall-clock (first row 2024-01-01T00:00 matches hourly file hour 0)
    d["ts"] = pd.to_datetime(d["Timestamp"].str[:19]).dt.floor("h")
    wd = np.deg2rad(pd.to_numeric(d["WD (deg)"], errors="coerce"))
    d["wd_sin"], d["wd_cos"] = np.sin(wd), np.cos(wd)
    d = d.rename(columns=RENAME)
    cols = list(RENAME.values()) + ["wd_sin", "wd_cos"]
    d[cols] = d[cols].apply(pd.to_numeric, errors="coerce")
    h = d.groupby(["site_id", "ts"])[cols].mean().reset_index()
    h["wd_deg"] = (np.degrees(np.arctan2(h["wd_sin"], h["wd_cos"])) + 360) % 360
    parts.append(h.drop(columns=["wd_sin", "wd_cos"]))
met = pd.concat(parts)

# ---- 3. registry + merge
reg = pd.read_csv(sys.argv[3] if len(sys.argv) > 3 else f"{SRC}/stations.csv")
reg = reg.dropna(subset=["latitude", "longitude"]).copy()
reg["key"] = reg["station_name"].str.split(",").str[0].str.strip()
SITE = {"Ballygunge": 5238, "Bidhannagar": 5129, "Fort William": 5110, "Rabindra Bharati University": 296,
        "Jadavpur": 5111, "Rabindra Sarobar": 5126, "Victoria": 309, "Belur Math": 5101, "Ghusuri": 274, "Padmapukur": 1416}
reg["site_id"] = reg["key"].map(SITE)
print("excluded from registry (no CPCB site id / not Kolkata):", reg.loc[reg.site_id.isna(), "key"].tolist())
reg = reg.dropna(subset=["site_id"]).astype({"site_id": int})
reg[["site_id", "key", "latitude", "longitude", "station_name"]].rename(columns={"key": "station"}).to_csv(f"{OUT}/station_registry.csv", index=False)

full = aqi.merge(met, on=["site_id", "ts"], how="left").merge(reg[["site_id", "key", "latitude", "longitude"]], on="site_id")
full = full.rename(columns={"key": "station"})
full.to_csv(f"{OUT}/hourly_all_stations.csv", index=False)

q = full.groupby("station").agg(hours=("ts", "count"), first=("ts", "min"), last=("ts", "max"),
        aqi_fill=("aqi", lambda s: round(100 * s.notna().mean(), 1)),
        pm25_fill=("pm25", lambda s: round(100 * s.notna().mean(), 1)),
        wd_fill=("wd_deg", lambda s: round(100 * s.notna().mean(), 1)))
q.to_csv(f"{OUT}/quality_report.csv"); print(q.to_string())
