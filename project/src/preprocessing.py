"""Inspection, cleaning, and merging functions."""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

from src.utils import (
    coerce_numeric_columns,
    find_first_matching_column,
    get_logger,
    normalize_columns,
    parse_datetime_series,
    strip_string_values,
    summarize_dataframe,
)


LOGGER = get_logger(__name__)


def inspect_dataframe(
    df: pd.DataFrame,
    *,
    name: str,
    important_columns: Sequence[str] | None = None,
) -> dict[str, object]:
    """Return standard initial-inspection outputs for a dataframe."""
    LOGGER.info("Inspecting %s", name)
    summary = summarize_dataframe(df, important_columns or [])
    return {
        "head": df.head(),
        "tail": df.tail(),
        "info": _dataframe_info(df),
        "describe": df.describe(include="all"),
        **summary,
    }


def _dataframe_info(df: pd.DataFrame) -> pd.DataFrame:
    """Represent dataframe info as a dataframe-friendly object."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "non_null_count": [df[column].notna().sum() for column in df.columns],
            "dtype": [df[column].dtype for column in df.columns],
        }
    )


def clean_trader_data(
    df: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Clean historical trader data while preserving all usable records."""
    active_logger = logger or LOGGER
    output = normalize_columns(df)
    active_logger.info("Normalized trader column names.")

    before_rows = len(output)
    output = output.drop_duplicates().reset_index(drop=True)
    active_logger.info("Removed duplicate trader rows: %s", before_rows - len(output))

    output = strip_string_values(output)
    active_logger.info("Stripped whitespace from trader text columns.")

    timestamp_column = find_first_matching_column(
        output.columns,
        ["timestamp_ist", "timestamp", "time", "datetime", "created_at", "date"],
    )
    output[timestamp_column] = parse_datetime_series(output[timestamp_column])
    invalid_timestamps = output[timestamp_column].isna().sum()
    active_logger.info("Converted trader timestamp column: %s", timestamp_column)
    if invalid_timestamps:
        active_logger.info("Trader rows with invalid timestamps: %s", invalid_timestamps)

    candidate_numeric_columns = [
        column
        for column in output.columns
        if any(
            token in column
            for token in [
                "price",
                "size",
                "usd",
                "pnl",
                "fee",
                "position",
                "leverage",
                "qty",
                "quantity",
                "amount",
            ]
        )
    ]
    output = coerce_numeric_columns(output, candidate_numeric_columns, active_logger)

    output = output.dropna(subset=[timestamp_column]).reset_index(drop=True)
    active_logger.info("Dropped trader rows missing valid timestamps.")
    return output


def clean_sentiment_data(
    df: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Clean Fear & Greed data and prepare a daily merge key."""
    active_logger = logger or LOGGER
    output = normalize_columns(df)
    active_logger.info("Normalized sentiment column names.")

    before_rows = len(output)
    output = output.drop_duplicates().reset_index(drop=True)
    active_logger.info("Removed duplicate sentiment rows: %s", before_rows - len(output))

    output = strip_string_values(output)
    active_logger.info("Stripped whitespace from sentiment text columns.")

    date_column = find_first_matching_column(
        output.columns,
        ["date", "timestamp", "time", "datetime"],
    )
    output[date_column] = parse_datetime_series(output[date_column])
    output = output.dropna(subset=[date_column]).reset_index(drop=True)
    output["trade_date"] = output[date_column].dt.normalize()
    active_logger.info("Converted sentiment date column and created trade_date.")

    value_column = find_first_matching_column(
        output.columns,
        ["value", "score", "fear_greed"],
        required=False,
    )
    if value_column:
        output = coerce_numeric_columns(output, [value_column], active_logger)

    return output


def merge_trader_with_sentiment(
    trader_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Attach daily sentiment data to each trade."""
    active_logger = logger or LOGGER
    output_trader = trader_df.copy()

    timestamp_column = find_first_matching_column(
        output_trader.columns,
        ["timestamp_ist", "timestamp", "time", "datetime", "created_at", "date"],
    )
    output_trader["trade_date"] = parse_datetime_series(
        output_trader[timestamp_column]
    ).dt.normalize()

    if "trade_date" not in sentiment_df.columns:
        raise KeyError("Sentiment data must contain trade_date before merging.")

    sentiment_daily = sentiment_df.drop_duplicates(subset=["trade_date"])
    merged = output_trader.merge(
        sentiment_daily,
        on="trade_date",
        how="left",
        suffixes=("", "_sentiment"),
    )

    if len(merged) != len(output_trader):
        raise ValueError(
            "Merged row count does not match trader row count: "
            f"{len(merged)} != {len(output_trader)}"
        )

    sentiment_columns = merged.filter(regex="classification|value|score")
    unmatched_rows = (
        sentiment_columns.isna().all(axis=1).sum() if not sentiment_columns.empty else 0
    )
    active_logger.info("Merged trader rows with daily sentiment: %s rows", len(merged))
    active_logger.info("Trades without matched sentiment fields: %s", unmatched_rows)
    return merged
