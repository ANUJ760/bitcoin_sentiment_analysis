"""Feature engineering helpers for merged trading and sentiment data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils import find_first_matching_column, get_logger, parse_datetime_series


LOGGER = get_logger(__name__)

SENTIMENT_SCORE_MAP = {
    "extreme fear": 1,
    "fear": 2,
    "neutral": 3,
    "greed": 4,
    "extreme greed": 5,
}


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create date and calendar features from the trade timestamp."""
    output = df.copy()
    timestamp_column = find_first_matching_column(
        output.columns,
        ["timestamp_ist", "timestamp", "time", "datetime", "created_at", "date"],
    )
    timestamps = parse_datetime_series(output[timestamp_column])
    output["trade_date"] = timestamps.dt.date
    output["trade_hour"] = timestamps.dt.hour
    output["trade_weekday"] = timestamps.dt.day_name()
    output["trade_month"] = timestamps.dt.month
    output["trade_year"] = timestamps.dt.year
    return output


def add_profitability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create profit, loss, win, absolute PnL, and return-per-unit features."""
    output = df.copy()
    pnl_column = find_first_matching_column(
        output.columns,
        ["closed_pnl", "pnl", "profit", "realized_pnl"],
    )
    output[pnl_column] = pd.to_numeric(output[pnl_column], errors="coerce")

    output["profit"] = output[pnl_column].where(output[pnl_column] > 0, 0)
    output["loss"] = output[pnl_column].where(output[pnl_column] < 0, 0).abs()
    output["win"] = output[pnl_column] > 0
    output["absolute_pnl"] = output[pnl_column].abs()

    size_column = find_first_matching_column(
        output.columns,
        ["size_usd", "trade_size", "notional", "amount_usd", "usd"],
        required=False,
    )
    if size_column:
        denominator = pd.to_numeric(output[size_column], errors="coerce").abs()
        output["return_per_unit"] = np.where(
            denominator > 0,
            output[pnl_column] / denominator,
            np.nan,
        )
    else:
        output["return_per_unit"] = np.nan

    return output


def add_bucket_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leverage and trade-size bucket features when source columns exist."""
    output = df.copy()

    leverage_column = find_first_matching_column(
        output.columns,
        ["leverage", "lev"],
        required=False,
    )
    if leverage_column:
        leverage = pd.to_numeric(output[leverage_column], errors="coerce")
        output["leverage_bucket"] = pd.cut(
            leverage,
            bins=[-np.inf, 1, 3, 5, 10, 20, np.inf],
            labels=["1x_or_less", "2x_3x", "4x_5x", "6x_10x", "11x_20x", "20x_plus"],
        )
    else:
        output["leverage_bucket"] = "unknown"

    size_column = find_first_matching_column(
        output.columns,
        ["size_usd", "trade_size", "notional", "amount_usd", "usd"],
        required=False,
    )
    if size_column:
        size = pd.to_numeric(output[size_column], errors="coerce").abs()
        if size.notna().sum() >= 4:
            output["trade_size_bucket"] = pd.qcut(
                size.rank(method="first"),
                q=4,
                labels=["small", "medium", "large", "very_large"],
                duplicates="drop",
            )
        else:
            output["trade_size_bucket"] = "insufficient_data"
    else:
        output["trade_size_bucket"] = "unknown"

    return output


def add_sentiment_score(df: pd.DataFrame) -> pd.DataFrame:
    """Map sentiment classification labels to ordinal sentiment scores."""
    output = df.copy()
    classification_column = find_first_matching_column(
        output.columns,
        ["classification", "sentiment", "fear_greed_classification"],
        required=False,
    )

    if classification_column:
        normalized = output[classification_column].astype("string").str.strip().str.lower()
        output["sentiment_score"] = normalized.map(SENTIMENT_SCORE_MAP)
    else:
        output["sentiment_score"] = np.nan

    return output


def engineer_features(
    df: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Run all feature engineering steps for the merged dataset."""
    active_logger = logger or LOGGER
    output = add_temporal_features(df)
    active_logger.info("Created temporal features.")

    output = add_profitability_features(output)
    active_logger.info("Created profitability features.")

    output = add_bucket_features(output)
    active_logger.info("Created leverage and trade-size bucket features.")

    output = add_sentiment_score(output)
    active_logger.info("Created sentiment_score feature.")

    return output
