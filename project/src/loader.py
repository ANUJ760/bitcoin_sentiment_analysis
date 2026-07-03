"""Dataset download and loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import GOOGLE_DRIVE_DOWNLOAD_URL
from src.utils import get_logger


LOGGER = get_logger(__name__)


def is_colab_runtime() -> bool:
    """Return True when the code is running in Google Colab."""
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def download_google_drive_file(file_id: str, output_path: Path, timeout: int = 60) -> Path:
    """Download a public Google Drive file to the requested path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = GOOGLE_DRIVE_DOWNLOAD_URL.format(file_id=file_id)

    LOGGER.info("Attempting download from Google Drive: %s", output_path.name)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "text/html" in content_type and b"Google Drive" in response.content[:5000]:
        raise RuntimeError(
            "Google Drive returned an HTML page instead of a data file. "
            "Manual upload may be required."
        )

    output_path.write_bytes(response.content)
    if output_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {output_path}")

    LOGGER.info("Downloaded dataset to %s", output_path)
    return output_path


def upload_file_in_colab(data_dir: Path) -> list[Path]:
    """Upload files in Google Colab and save them to the project data directory."""
    if not is_colab_runtime():
        raise RuntimeError("Colab upload fallback is only available in Google Colab.")

    from google.colab import files  # type: ignore

    data_dir.mkdir(parents=True, exist_ok=True)
    uploaded: dict[str, bytes] = files.upload()
    saved_paths: list[Path] = []

    for filename, content in uploaded.items():
        destination = data_dir / Path(filename).name
        destination.write_bytes(content)
        saved_paths.append(destination)
        LOGGER.info("Uploaded dataset saved to %s", destination)

    return saved_paths


def ensure_dataset(
    *,
    file_id: str,
    default_filename: str,
    data_dir: Path,
    dataset_label: str,
) -> Path:
    """Ensure a dataset exists locally, trying download before Colab upload."""
    data_dir.mkdir(parents=True, exist_ok=True)
    existing_files = sorted(data_dir.glob("*"))
    matching_existing = [
        path
        for path in existing_files
        if path.is_file() and dataset_label.lower() in path.stem.lower()
    ]
    if matching_existing:
        LOGGER.info("Using existing dataset for %s: %s", dataset_label, matching_existing[0])
        return matching_existing[0]

    output_path = data_dir / default_filename
    if output_path.exists():
        LOGGER.info("Using existing dataset: %s", output_path)
        return output_path

    try:
        return download_google_drive_file(file_id, output_path)
    except Exception as exc:
        LOGGER.warning("Automatic download failed for %s: %s", dataset_label, exc)
        if is_colab_runtime():
            uploaded_paths = upload_file_in_colab(data_dir)
            if not uploaded_paths:
                raise FileNotFoundError("No files were uploaded in Colab.") from exc
            return uploaded_paths[0]
        raise FileNotFoundError(
            f"Could not download {dataset_label}. Place the file in {data_dir} "
            "or run the notebook in Colab and upload it manually."
        ) from exc


def load_tabular_file(path: Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load a CSV or Excel file into a dataframe."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, **read_csv_kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    try:
        return pd.read_csv(path, **read_csv_kwargs)
    except Exception as exc:
        raise ValueError(f"Unsupported or unreadable dataset format: {path}") from exc
