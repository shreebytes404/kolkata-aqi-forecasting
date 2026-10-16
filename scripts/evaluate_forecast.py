"""Expanding-window evaluation of multi-horizon forecasts vs persistence.
usage: python evaluate_forecast.py data/app/hourly_aqi.csv.gz outdir"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/src")
from forecast import HORIZONS, feature_cols, fit, make_frame

df = pd.read_csv(sys.argv[1], parse_dates=["ts"]); out = sys.argv[2]
wide = df.pivot(index="ts", columns="station", values="aqi")
FOLDS = [("2025-01-01", "2025-03-31"), ("2025-07-01", "2025-09-30"),
         ("2025-10-01", "2026-01-31"), ("2026-02-01", "2026-09-30")]
rows = []
for h in HORIZONS:
    for lags in (False, True):
        for a, b in FOLDS:
            a, b = pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(hours=23)
            m = fit(wide, h, lags, until=a)
            te = make_frame(wide, h, lags).dropna(subset=["aqi_now", "target"])
            te = te[(te.ts >= a) & (te.ts <= b)]
            p = np.clip(te.aqi_now + m.predict(te[feature_cols(lags)]), 0, 500)
            rows.append(dict(h=h, variant="full" if lags else "light", fold=f"{a:%Y-%m}..{b:%Y-%m}", n=len(te),
                             mae_model=np.abs(p - te.target).mean(), mae_persist=np.abs(te.aqi_now - te.target).mean()))
r = pd.DataFrame(rows); r.to_csv(f"{out}/forecast_folds.csv", index=False)
tot = r.assign(wm=r.mae_model * r.n, wp=r.mae_persist * r.n).groupby(["h", "variant"]).agg(n=("n", "sum"), wm=("wm", "sum"), wp=("wp", "sum"))
tot["MAE_model"] = (tot.wm / tot.n).round(2); tot["MAE_persistence"] = (tot.wp / tot.n).round(2)
tot["improvement_%"] = (100 * (1 - tot.MAE_model / tot.MAE_persistence)).round(1)
tot = tot[["n", "MAE_model", "MAE_persistence", "improvement_%"]].reset_index(); tot.to_csv(f"{out}/forecast_summary.csv", index=False)
print(tot.to_string(index=False)); print(r.round(2).to_string(index=False))
