"""Project configuration and filesystem paths."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TRADER_FILE_ID = "1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs"
SENTIMENT_FILE_ID = "1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf"

GOOGLE_DRIVE_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id={file_id}"

TRADER_DATASET = {
    "name": "historical_trader_data",
    "file_id": TRADER_FILE_ID,
    "default_filename": "historical_trader_data.csv",
}

SENTIMENT_DATASET = {
    "name": "fear_greed_index",
    "file_id": SENTIMENT_FILE_ID,
    "default_filename": "fear_greed_index.csv",
}

REQUIRED_DIRECTORIES = (DATA_DIR, FIGURES_DIR, OUTPUTS_DIR)
