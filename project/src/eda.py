"""Exploratory data analysis helpers and plotting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

from src.utils import find_first_matching_column


WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


@dataclass(frozen=True)
class EDAColumns:
    """Column map used by the EDA helpers."""

    pnl: str
    leverage: str | None
    trade_size: str | None
    account: str | None
    symbol: str | None
    side: str | None
    sentiment: str | None
    sentiment_score: str | None


def configure_plot_style() -> None:
    """Apply a professional plotting style for the notebook."""
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette="deep",
        font_scale=1.0,
    )
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.titleweight": "bold",
            "axes.labelweight": "regular",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "figure.figsize": (10, 5),
        }
    )


def get_eda_columns(df: pd.DataFrame) -> EDAColumns:
    """Infer important EDA columns from the feature-engineered dataset."""
    leverage_column = find_first_matching_column(
        df.columns,
        ["leverage", "lev"],
        required=False,
    )
    if leverage_column and "bucket" in leverage_column.lower():
        leverage_column = None

    return EDAColumns(
        pnl=find_first_matching_column(
            df.columns,
            ["closed_pnl", "pnl", "realized_pnl", "profit"],
        ),
        leverage=leverage_column,
        trade_size=find_first_matching_column(
            df.columns,
            ["size_usd", "trade_size", "notional", "amount_usd", "usd"],
            required=False,
        ),
        account=find_first_matching_column(
            df.columns,
            ["account", "account_address", "trader", "user", "wallet"],
            required=False,
        ),
        symbol=find_first_matching_column(
            df.columns,
            ["coin", "symbol", "asset", "market", "token"],
            required=False,
        ),
        side=find_first_matching_column(
            df.columns,
            ["side", "direction", "buy_sell", "order_side"],
            required=False,
        ),
        sentiment=find_first_matching_column(
            df.columns,
            ["classification", "sentiment", "fear_greed_classification"],
            required=False,
        ),
        sentiment_score=find_first_matching_column(
            df.columns,
            ["sentiment_score"],
            required=False,
        ),
    )


def save_current_figure(figures_dir: Path, filename: str) -> Path:
    """Save the current matplotlib figure with project-standard settings."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def show_observation(
    *,
    observation: str,
    interpretation: str,
    business_insight: str,
) -> None:
    """Display chart commentary in the required markdown format."""
    display(
        Markdown(
            f"**Observation:** {observation}\n\n"
            f"**Interpretation:** {interpretation}\n\n"
            f"**Business Insight:** {business_insight}"
        )
    )


def require_column(column: str | None, purpose: str) -> str:
    """Raise a clear error when a required analysis column is unavailable."""
    if column is None:
        raise KeyError(f"Cannot complete {purpose}; required source column was not found.")
    return column


def get_dataset_overview(df: pd.DataFrame, columns: EDAColumns) -> pd.DataFrame:
    """Compute the requested dataset overview metrics."""
    pnl = pd.to_numeric(df[columns.pnl], errors="coerce")
    leverage = pd.to_numeric(df[columns.leverage], errors="coerce") if columns.leverage else None
    time_span = (
        f"{df['trade_date'].min()} to {df['trade_date'].max()}"
        if "trade_date" in df.columns
        else "trade_date unavailable"
    )
    trade_frequency = (
        df.groupby("trade_date").size().mean() if "trade_date" in df.columns else np.nan
    )

    metrics = {
        "Total trades": len(df),
        "Number of traders": df[columns.account].nunique() if columns.account else np.nan,
        "Unique symbols": df[columns.symbol].nunique() if columns.symbol else np.nan,
        "Time span": time_span,
        "Average leverage": leverage.mean() if leverage is not None else np.nan,
        "Median leverage": leverage.median() if leverage is not None else np.nan,
        "Average PnL": pnl.mean(),
        "Median PnL": pnl.median(),
        "Maximum profit": pnl.max(),
        "Maximum loss": pnl.min(),
        "Average trades per day": trade_frequency,
    }
    return pd.DataFrame({"Metric": metrics.keys(), "Value": metrics.values()})


def plot_count(
    df: pd.DataFrame,
    column: str,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    figures_dir: Path,
    order: Iterable[str] | None = None,
    top_n: int | None = None,
    rotate: int = 0,
) -> pd.Series:
    """Plot and save a count bar chart."""
    counts = df[column].value_counts(dropna=False)
    if top_n:
        counts = counts.head(top_n)
    if order:
        counts = counts.reindex([value for value in order if value in counts.index])

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=counts.index.astype(str), y=counts.values, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=rotate, ha="right" if rotate else "center")
    save_current_figure(figures_dir, filename)
    plt.show()
    return counts


def plot_group_metric(
    df: pd.DataFrame,
    *,
    group_column: str,
    value_column: str,
    metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    figures_dir: Path,
    sort_values: bool = True,
    top_n: int | None = None,
    rotate: int = 0,
) -> pd.Series:
    """Plot and save a grouped aggregate bar chart."""
    values = pd.to_numeric(df[value_column], errors="coerce")
    grouped = getattr(values.groupby(df[group_column]), metric)()
    if sort_values:
        grouped = grouped.sort_values(ascending=False)
    if top_n:
        grouped = grouped.head(top_n)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=grouped.index.astype(str), y=grouped.values, color="#59A14F")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=rotate, ha="right" if rotate else "center")
    save_current_figure(figures_dir, filename)
    plt.show()
    return grouped


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    *,
    title: str,
    xlabel: str,
    filename: str,
    figures_dir: Path,
    kde: bool = False,
) -> pd.Series:
    """Plot and save a numeric histogram."""
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    plt.figure(figsize=(10, 5))
    ax = sns.histplot(values, bins=50, kde=kde, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Trade Count")
    ax.grid(axis="y", alpha=0.3)
    save_current_figure(figures_dir, filename)
    plt.show()
    return values


def plot_box(
    df: pd.DataFrame,
    *,
    x: str | None,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    figures_dir: Path,
    violin: bool = False,
    rotate: int = 0,
) -> None:
    """Plot and save a boxplot or violin plot."""
    plt.figure(figsize=(11, 5))
    plotter: Callable[..., object] = sns.violinplot if violin else sns.boxplot
    if x:
        plotter(data=df, x=x, y=y, color="#F28E2B")
    else:
        plotter(y=pd.to_numeric(df[y], errors="coerce"), color="#F28E2B")
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=rotate, ha="right" if rotate else "center")
    save_current_figure(figures_dir, filename)
    plt.show()


def plot_scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    figures_dir: Path,
    regression: bool = False,
) -> float:
    """Plot and save a scatter plot, optionally with a regression line."""
    plot_df = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    plt.figure(figsize=(10, 5))
    if regression and len(plot_df) >= 3:
        ax = sns.regplot(data=plot_df, x=x, y=y, scatter_kws={"alpha": 0.35})
    else:
        ax = sns.scatterplot(data=plot_df, x=x, y=y, alpha=0.45)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    save_current_figure(figures_dir, filename)
    plt.show()
    return plot_df[x].corr(plot_df[y]) if len(plot_df) >= 2 else np.nan


def display_top_bottom(
    df: pd.DataFrame,
    *,
    sort_column: str,
    columns: list[str],
    n: int,
    ascending: bool,
    title: str,
) -> pd.DataFrame:
    """Display a top or bottom ranked table."""
    table = df.sort_values(sort_column, ascending=ascending).loc[:, columns].head(n)
    display(Markdown(f"**{title}**"))
    display(table)
    return table


def win_rate_by_group(df: pd.DataFrame, group_column: str) -> pd.Series:
    """Compute win rate by a categorical grouping column."""
    if "win" not in df.columns:
        raise KeyError("win column is required for win-rate analysis.")
    return df.groupby(group_column)["win"].mean().sort_values(ascending=False)


def plot_win_rate(
    df: pd.DataFrame,
    group_column: str,
    *,
    title: str,
    filename: str,
    figures_dir: Path,
    top_n: int | None = None,
    rotate: int = 0,
) -> pd.Series:
    """Plot and save win rate by group."""
    rates = win_rate_by_group(df, group_column)
    if top_n:
        rates = rates.head(top_n)
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=rates.index.astype(str), y=rates.values, color="#76B7B2")
    ax.set_title(title)
    ax.set_xlabel(group_column.replace("_", " ").title())
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=rotate, ha="right" if rotate else "center")
    save_current_figure(figures_dir, filename)
    plt.show()
    return rates


def plot_positive_negative_stacked(
    df: pd.DataFrame,
    group_column: str,
    *,
    title: str,
    filename: str,
    figures_dir: Path,
) -> pd.DataFrame:
    """Plot stacked positive/negative trade counts by group."""
    stacked = (
        pd.crosstab(df[group_column], df["win"].map({True: "Positive", False: "Negative"}))
        .sort_index()
    )
    ax = stacked.plot(kind="bar", stacked=True, figsize=(10, 5), color=["#E15759", "#59A14F"])
    ax.set_title(title)
    ax.set_xlabel(group_column.replace("_", " ").title())
    ax.set_ylabel("Trade Count")
    ax.legend(title="Trade Outcome")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    save_current_figure(figures_dir, filename)
    plt.show()
    return stacked


def plot_timeline(
    df: pd.DataFrame,
    *,
    date_column: str,
    value_column: str,
    title: str,
    ylabel: str,
    filename: str,
    figures_dir: Path,
    aggregation: str = "mean",
) -> pd.Series:
    """Plot and save a daily timeline."""
    values = pd.to_numeric(df[value_column], errors="coerce")
    timeline = getattr(values.groupby(pd.to_datetime(df[date_column])), aggregation)()
    plt.figure(figsize=(12, 5))
    ax = sns.lineplot(x=timeline.index, y=timeline.values, marker="o", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    save_current_figure(figures_dir, filename)
    plt.show()
    return timeline


def plot_heatmap(
    data: pd.DataFrame,
    *,
    title: str,
    filename: str,
    figures_dir: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    """Plot and save a heatmap."""
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(data, cmap="viridis", linewidths=0.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_current_figure(figures_dir, filename)
    plt.show()


def describe_extreme(series: pd.Series, label: str) -> str:
    """Return a compact data-derived statement about the largest category/value."""
    clean = series.dropna()
    if clean.empty:
        return f"No {label} values were available."
    top_index = clean.idxmax()
    top_value = clean.max()
    return f"The highest {label} is {top_index} at {top_value:,.4g}."


def describe_correlation(correlation: float, x_label: str, y_label: str) -> str:
    """Return a data-derived correlation statement."""
    if pd.isna(correlation):
        return f"Not enough valid data was available to relate {x_label} and {y_label}."
    direction = "positive" if correlation > 0 else "negative" if correlation < 0 else "near-zero"
    return f"The Pearson correlation between {x_label} and {y_label} is {correlation:.3f}, a {direction} relationship in this dataset."


def create_correlation_matrix(
    df: pd.DataFrame,
    columns: EDAColumns,
) -> pd.DataFrame:
    """Create a numeric correlation matrix for requested EDA features."""
    selected = [columns.pnl, "trade_hour", "sentiment_score", "absolute_pnl", "return_per_unit"]
    if columns.leverage:
        selected.append(columns.leverage)
    if columns.trade_size:
        selected.append(columns.trade_size)
    if columns.account:
        trade_count_column = "account_trade_count"
        df = df.copy()
        df[trade_count_column] = df.groupby(columns.account)[columns.pnl].transform("size")
        selected.append(trade_count_column)
    available = [column for column in selected if column in df.columns]
    numeric_df = df[available].apply(pd.to_numeric, errors="coerce")
    return numeric_df.corr()
