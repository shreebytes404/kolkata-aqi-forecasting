"""Command-line entry point to build the model-ready dataset."""

from src.config import PROCESSED_PATH, RAW_AIR_PATH
from src.data_pipeline import build_features, fetch_daily_weather, load_air_quality


def main() -> None:
    air = load_air_quality(RAW_AIR_PATH)
    weather = fetch_daily_weather(air["date"].min(), air["date"].max())
    data = build_features(air, weather)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_PATH, index=False)
    missing_weather = data.filter(regex="temperature|humidity|wind|pressure").isna().mean().mean() * 100
    print(f"Saved {len(data)} rows to {PROCESSED_PATH}")
    print(f"Average weather-feature missingness: {missing_weather:.2f}%")


if __name__ == "__main__":
    main()
