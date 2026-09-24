"""Audit raw source files and build the model-ready four-station dataset."""

from __future__ import annotations

import json

from src.config import AUDIT_PATH, PROCESSED_PATH, RAW_DATA_DIR
from src.data_pipeline import (
    aggregate_to_daily,
    build_features,
    create_data_quality_report,
    fetch_daily_weather,
    read_raw_files,
    standardise_air_data,
)


def main() -> None:
    raw = read_raw_files(RAW_DATA_DIR)
    air = standardise_air_data(raw)
    audit = create_data_quality_report(air)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    daily = aggregate_to_daily(air)
    weather = fetch_daily_weather(daily["date"].min(), daily["date"].max())
    data = build_features(daily, weather)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_PATH, index=False)

    missing_weather = data.filter(regex="temperature|humidity|wind|pressure").isna().mean().mean() * 100
    print(f"Saved data-quality report to {AUDIT_PATH}")
    print(f"Saved {len(data)} model rows to {PROCESSED_PATH}")
    print(f"Average weather-feature missingness: {missing_weather:.2f}%")


if __name__ == "__main__":
    main()
