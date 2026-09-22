# Kolkata AQI forecasting

An auditable, software-only machine-learning project for forecasting **next-day AQI** at a Kolkata monitoring station. The first milestone uses Fort William; it can be extended to Rabindra Bharati University and Jadavpur after validation.

## Research question

Can historical station-level AQI and meteorological conditions predict next-day AQI more accurately than the persistence baseline ("tomorrow's AQI equals today's AQI")?

This is a short-term decision-support research prototype. It does **not** replace certified monitoring stations, issue official alerts, or establish causation.

## Workflow

1. Export/download verified historical daily AQI data from CPCB/WBPCB for the selected station. These are the preferred primary pollution-data sources for this conference paper.
2. Save it locally as `data/raw/air_quality_daily.csv`.
3. Fetch matching historical weather data from Open-Meteo Archive.
4. Build leakage-safe lag, rolling, weather, and calendar features.
5. Compare persistence, Linear Regression, and Random Forest with chronological evaluation.
6. Save the best evaluated model and show it in the Streamlit demo.

## Input-data contract

Create `data/raw/air_quality_daily.csv` with these exact columns:

```csv
date,station,aqi
2024-01-01,Fort William,126
2024-01-02,Fort William,118
```

- `date`: ISO date (`YYYY-MM-DD`), one record per daily AQI.
- `station`: exactly `Fort William` for the first milestone.
- `aqi`: documented daily AQI in the 0–500 range.
- Minimum: 120 valid daily rows; 18–24 months is preferred.

Keep the original export and a source log with URL, station identifier, download date, unit, AQI standard, and any data transformation. If a secondary public source is required because the official export is unavailable or incomplete, document its provider, access date, coverage, match to the named station, and limitation. Raw and processed data are deliberately excluded from Git to avoid publishing unverified or restricted data.

## Run locally or in Google Colab

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt

python -m src.prepare_data
python -m src.train
streamlit run app/streamlit_app.py
```

For Google Colab, upload this repository (or clone it), install `requirements.txt`, upload the CSV into `data/raw/`, and run the same two Python commands. Streamlit Cloud should use `app/streamlit_app.py` as its entry point after the trained model and permitted processed data are made available through a safe deployment method.

## Evaluation safeguards

- The last 20% of dates are a final untouched test set.
- `TimeSeriesSplit` evaluates models only on earlier-to-later time windows.
- Lag and rolling features only use data available at or before day *t* to predict day *t+1*.
- Median imputation happens inside the model pipeline, so validation/test values do not influence training transformations.
- Report MAE, RMSE, R-squared, cross-validated MAE, dates of the test period, and persistence-baseline results.

## Publication checklist

- Verify each claim and reference against its original source.
- State the AQI standard, averaging period, station selection rule, data cutoff date, and missing-data rate.
- Do not call a PM2.5-only target "official AQI." If complete pollutant data are unavailable, revise the paper target to PM2.5 or PM2.5-derived AQI.
- Add model limitations: station coverage, imputation, meteorological grid mismatch, seasonal shift, and lack of causal attribution.
- Include authors, mentor, institution, ethics/declaration text, and the required venue template only after mentor review.
- For a conference submission, add the conference name, its author instructions, anonymization rule (if any), page limit, citation style, and submission deadline before typesetting the final manuscript.

## Project layout

```text
data/raw/          original station export, ignored by Git
data/processed/    model dataset, ignored by Git
src/               preparation and training code
models/            saved model and evaluation metadata, ignored by Git
app/               Streamlit demo
reports/figures/   paper figures, ignored by Git
```
