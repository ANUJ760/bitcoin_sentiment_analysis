"""Shared utility functions for the analysis pipeline."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create required project directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "trading_sentiment") -> logging.Logger:
    """Return a configured logger without adding duplicate handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def normalize_column_name(column_name: object) -> str:
    """Convert a raw column name into snake_case."""
    normalized = str(column_name).strip().lower()
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a dataframe with normalized column names."""
    output = df.copy()
    output.columns = [normalize_column_name(column) for column in output.columns]
    return output


def strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading and trailing whitespace from object/string columns."""
    output = df.copy()
    text_columns = output.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        output[column] = output[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return output


def find_first_matching_column(
    columns: Sequence[str],
    candidates: Sequence[str],
    *,
    required: bool = True,
) -> str | None:
    """Find the first column containing any candidate token."""
    for candidate in candidates:
        normalized_candidate = normalize_column_name(candidate)
        for column in columns:
            normalized_column = normalize_column_name(column)
            if normalized_candidate == normalized_column:
                return column

    for candidate in candidates:
        normalized_candidate = normalize_column_name(candidate)
        for column in columns:
            normalized_column = normalize_column_name(column)
            if normalized_candidate in normalized_column:
                return column

    if required:
        raise KeyError(f"Unable to find a matching column for: {candidates}")
    return None


def coerce_numeric_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Convert selected columns to numeric values and log invalid counts."""
    output = df.copy()
    active_logger = logger or get_logger()
    for column in columns:
        if column not in output.columns:
            continue
        before_missing = output[column].isna().sum()
        output[column] = pd.to_numeric(output[column], errors="coerce")
        after_missing = output[column].isna().sum()
        invalid_count = max(after_missing - before_missing, 0)
        if invalid_count:
            active_logger.info(
                "Converted invalid numeric values to NaN in %s: %s",
                column,
                invalid_count,
            )
    return output


def parse_datetime_series(values: pd.Series) -> pd.Series:
    """Parse string or numeric timestamp values into pandas datetimes."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce")

    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_ratio = numeric_values.notna().mean()

    if numeric_ratio >= 0.8:
        median_value = numeric_values.dropna().abs().median()
        if pd.notna(median_value):
            if median_value >= 1_000_000_000_000:
                return pd.to_datetime(numeric_values, unit="ms", errors="coerce")
            if median_value >= 1_000_000_000:
                return pd.to_datetime(numeric_values, unit="s", errors="coerce")

    return pd.to_datetime(values, errors="coerce")


def summarize_dataframe(df: pd.DataFrame, important_columns: Sequence[str]) -> dict[str, object]:
    """Build an inspection summary without assuming display context."""
    unique_values = {
        column: df[column].dropna().unique()[:20].tolist()
        for column in important_columns
        if column in df.columns
    }
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isna().sum().sort_values(ascending=False),
        "duplicate_rows": int(df.duplicated().sum()),
        "data_types": df.dtypes,
        "unique_values": unique_values,
    }
