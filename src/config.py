"""Project-wide constants for the first, single-station MVP."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_AIR_PATH = ROOT / "data" / "raw" / "air_quality_daily.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "model_dataset.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

# Fort William is the initial station. Change only after documenting the reason.
DEFAULT_STATION = "Fort William"
KOLKATA_LATITUDE = 22.556
KOLKATA_LONGITUDE = 88.338
TARGET_COLUMN = "aqi_next_day"

# Only information available by the end of day t can be a model input for t+1.
LAGS = (1, 2, 3)
ROLLING_WINDOWS = (3, 7)
