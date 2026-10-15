"""Leave-one-station-out: hide each station, estimate its AQI from the others, compare methods.
usage: python evaluate_loso.py hourly_all_stations.csv outdir
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/src")
from spatial import estimate, circular_mean_deg

df = pd.read_csv(sys.argv[1], parse_dates=["ts"]); out = sys.argv[2]
aqi = df.pivot(index="ts", columns="station", values="aqi")
wd = df.pivot(index="ts", columns="station", values="wd_deg")
reg = df.drop_duplicates("station").set_index("station")[["latitude", "longitude"]]
st = list(aqi.columns)
city_wd = wd.apply(lambda r: circular_mean_deg(r.values), axis=1)
hours = aqi.index[city_wd.notna()]                      # hours with an observed wind (2024-25)
hours = hours[::3]                                      # every 3rd hour: cuts compute, hours are highly autocorrelated anyway
A, W = aqi.loc[hours].values, city_wd.loc[hours].values
lat, lon = reg.loc[st, "latitude"].values, reg.loc[st, "longitude"].values

METHODS = {"nearest": None, "idw": 0.0, "wind_0.5": 0.5, "wind_1.5": 1.5, "wind_4": 4.0}
rows = []
for j, s in enumerate(st):
    src = [k for k in range(len(st)) if k != j]
    for i in range(len(hours)):
        truth = A[i, j]
        if np.isnan(truth): continue
        a = A[i, src]
        if np.isnan(a).all(): continue
        rec = {"station": s, "ts": hours[i], "truth": truth}
        dk = None
        for name, strength in METHODS.items():
            if name == "nearest":
                _, dk, _ = estimate(lat[j], lon[j], lat[src], lon[src], a)
                dd = np.where(np.isnan(a), np.inf, [np.hypot(lat[k]-lat[j], lon[k]-lon[j]) for k in src])
                rec[name] = a[int(np.argmin(dd))]
            else:
                rec[name], dk, _ = estimate(lat[j], lon[j], lat[src], lon[src], a, W[i], wind_strength=strength)
        rec["nearest_km"] = dk
        rows.append(rec)
r = pd.DataFrame(rows); r.to_csv(f"{out}/loso_predictions.csv", index=False)
res = []
for m in METHODS:
    e = r[m] - r["truth"]
    res.append((m, e.abs().mean(), np.sqrt((e**2).mean()), len(r)))
summ = pd.DataFrame(res, columns=["method", "MAE", "RMSE", "n"]).round(2); print(summ.to_string(index=False))
summ.to_csv(f"{out}/loso_summary.csv", index=False)
by = r.groupby("station").apply(lambda g: pd.Series({m: (g[m]-g["truth"]).abs().mean() for m in METHODS})).round(2)
by.to_csv(f"{out}/loso_by_station.csv"); print(by.to_string())
r["season"] = np.where(r.ts.dt.month.isin([11,12,1,2]), "winter", np.where(r.ts.dt.month.isin([6,7,8,9]), "monsoon", "other"))
print(r.groupby("season").apply(lambda g: pd.Series({m: (g[m]-g["truth"]).abs().mean() for m in METHODS})).round(2).to_string())
