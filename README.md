# Bitcoin Market Sentiment vs Hyperliquid Trader Performance Analysis

## Project Overview

This repository analyzes how Bitcoin market sentiment, measured by the Fear & Greed Index, relates to Hyperliquid trader performance. The project combines trade-level outcomes with daily sentiment context to study profitability, trade size, timing, and regime differences in a reproducible notebook workflow.

## Objective

The goal is to answer evidence-based questions such as:

- Does profitability differ across sentiment regimes?
- Are larger trades associated with larger PnL outcomes?
- Does sentiment show a measurable relationship with trade performance?
- Which simple features are useful for exploratory forecasting?

All conclusions in the notebook and report are derived directly from the data.

## Repository Structure

```text
project/
|-- analysis.ipynb
|-- report.md
|-- README.md
|-- requirements.txt
|-- data/
|-- figures/
|-- outputs/
|-- src/
|   |-- config.py
|   |-- eda.py
|   |-- feature_engineering.py
|   |-- loader.py
|   |-- preprocessing.py
|   `-- utils.py
```

## Datasets

- Historical Trader Data: Hyperliquid trade history used for PnL, trade size, timing, and account-level analysis.
- Fear & Greed Dataset: Daily Bitcoin sentiment index used to attach a market regime to each trade day.

The notebook downloads both datasets automatically from Google Drive when possible. If a download is blocked, the loader falls back to manual upload in Colab. Local runs read the files from `data/`.

## Methodology

1. Load and inspect the raw datasets.
2. Clean and standardize column names, timestamps, numeric fields, and merge keys.
3. Merge daily sentiment onto trade-level records.
4. Engineer temporal, profitability, size, and sentiment features.
5. Perform exploratory data analysis and save figures to `figures/`.
6. Run statistical tests for normality, variance, correlations, and sentiment-group differences.
7. Fit a lightweight exploratory predictive model for closed PnL.
8. Summarize findings, limitations, recommendations, and future work.

## Feature Engineering

The notebook creates reusable features including:

- `trade_date`, `trade_hour`, `trade_weekday`, `trade_month`, `trade_year`
- `profit`, `loss`, `win`, `absolute_pnl`
- `trade_size_bucket`
- `sentiment_score`
- `return_per_unit` when trade size is available

## Exploratory Data Analysis

The notebook covers:

- Dataset overview metrics
- Sentiment distribution and sentiment-over-time views
- PnL distributions and outlier inspection
- Win-rate comparisons by sentiment, symbol, and account
- Trade size and time-of-day analysis
- Correlation heatmaps and pairwise numeric comparisons
- Saved figures in `figures/` with descriptive filenames

## Statistical Analysis

The final notebook section includes:

- Shapiro-Wilk normality checks for key numeric variables
- Levene variance testing across sentiment groups
- Pearson and Spearman correlations
- Kruskal-Wallis group comparison across sentiment regimes
- Holm-corrected post-hoc Mann-Whitney comparisons when overall significance exists
- A lightweight Random Forest regression benchmark for exploratory prediction of `closed_pnl`

## Optional Modeling

The exploratory model uses a train/test split with `random_state=42` and reports MAE, RMSE, and R². It is included to benchmark simple predictive signal, not to imply trading viability.

## Key Findings

- Profitability differed significantly across sentiment groups in the matched sample.
- Extreme Greed had the highest mean PnL and win rate among the matched sentiment groups.
- Extreme Fear had the weakest mean PnL and the lowest win rate.
- Greed and Neutral were not significantly different after Holm-corrected post-hoc testing.
- Closed PnL had a weak positive relationship with trade size and a very weak positive relationship with sentiment score.
- The exploratory Random Forest model was modest, with R² around 0.24, so it should be treated as a baseline only.

## Visual Examples

Representative figures saved in `figures/` include:

- `average_pnl_by_sentiment.png`
- `trade_count_by_sentiment.png`
- `pnl_histogram.png`
- `win_rate_by_sentiment.png`
- `trading_activity_heatmap.png`
- `correlation_matrix_heatmap.png`

## Installation

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Open `analysis.ipynb` in Jupyter or VS Code.
2. Run the notebook from top to bottom.
3. Review the generated figures in `figures/` and the written report in `report.md`.

## Requirements

The project uses:

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
- statsmodels
- requests
- jupyter

## Future Improvements

- Add richer market features such as funding rates, liquidation data, and order-book context.
- Expand the analysis with walk-forward validation and regime-aware modeling.
- Add trader clustering if more complete risk features become available.
- Compare results across different market windows to check for stability.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## Acknowledgements

- Hyperliquid trader data and the Bitcoin Fear & Greed Index dataset used in the analysis.
- The notebook and helper modules in `src/` were structured to support reproducible data-science review.


