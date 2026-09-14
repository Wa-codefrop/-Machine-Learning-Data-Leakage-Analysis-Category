"""Project configuration values used across the repository."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "seo_content_performance.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "seo_content_performance_processed.csv"
EXPERIMENTS_PATH = PROJECT_ROOT / "experiments" / "results.csv"

RANDOM_STATE = 42
TARGET_COLUMN = "is_declining_label"
LEAKAGE_COLUMNS = [
    "trend_direction",
    "trend_pct",
    "future_clicks",
    "future_position",
]
