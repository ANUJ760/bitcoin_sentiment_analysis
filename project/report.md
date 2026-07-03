# Executive Summary

This project examines how Bitcoin market sentiment relates to Hyperliquid trader performance. Using 79,225 cleaned trades and a daily Fear & Greed sentiment merge, the analysis finds that profitability is not evenly distributed across sentiment regimes. The strongest matched-sample performance appears in Extreme Greed, while Extreme Fear shows the weakest profitability and win rate. Statistical testing confirms that the sentiment groups are different overall, although Greed and Neutral are not significantly different after Holm-corrected post-hoc testing.

The analysis also shows that closed PnL has a weak positive relationship with trade size and a very weak positive relationship with sentiment score. A lightweight Random Forest model provides a modest exploratory benchmark, but its R² is only about 0.24, so it should not be treated as production trading advice.

# Introduction

This report summarizes an exploratory data analysis project that combines Hyperliquid trading records with the Bitcoin Fear & Greed Index. The goal is to determine whether market sentiment is associated with trade profitability, risk behavior, and simple predictive signals.

# Problem Statement

The project asks whether trader outcomes differ across market sentiment regimes and whether simple engineered features can help explain profitability. The key business question is not whether sentiment predicts prices perfectly, but whether sentiment aligns with measurable differences in trader behavior and outcome distribution.

# Datasets

The analysis uses two sources:

| Dataset | Purpose |
|---|---|
| Historical Trader Data | Trade-level outcomes, size, timing, side, and account information |
| Fear & Greed Dataset | Daily market sentiment regime attached to each trade date |

After cleaning, the trader dataset contains 79,225 valid trade rows. Sentiment was successfully matched to 35,864 of those trades for sentiment-based statistical analysis.

# Methodology

1. Load both datasets from Google Drive or local storage.
2. Inspect raw schemas and data quality.
3. Clean timestamps, strip text columns, coerce numeric fields, and remove duplicate or invalid rows.
4. Merge daily sentiment to trade-level records.
5. Engineer temporal, profitability, bucket, and sentiment features.
6. Run exploratory visual analysis.
7. Conduct formal statistical tests.
8. Fit a lightweight exploratory predictive model.

# Cleaning

The raw trader file contained many invalid timestamps, which were dropped during cleaning. The cleaned trader dataset retained 79,225 valid rows. The sentiment dataset was normalized to a daily merge key so each trade could be paired with the sentiment state for that day.

# Feature Engineering

The notebook creates the following features:

- `trade_date`, `trade_hour`, `trade_weekday`, `trade_month`, `trade_year`
- `profit`, `loss`, `win`, `absolute_pnl`
- `return_per_unit`
- `trade_size_bucket`
- `sentiment_score`

These features support descriptive analysis, group comparisons, and the lightweight exploratory model.

# Exploratory Data Analysis

The EDA section examines sentiment composition, PnL distribution, trade activity through time, size behavior, and account-level patterns. The project saves figures to `figures/` using descriptive file names so the notebook can be reviewed as a reproducible analysis artifact.

Notable descriptive metrics from the cleaned trade set include:

| Metric | Value |
|---|---:|
| Trades | 79,225 |
| Mean closed PnL | 71.68 |
| Median closed PnL | 0.00 |
| Mean trade size | 5,842.28 |
| Median trade size | 662.02 |

The PnL distribution is highly skewed, with both large gains and large losses, which makes robust and non-parametric analysis important.

# Statistical Analysis

## Normality

Shapiro-Wilk tests on sampled observations rejected normality for all evaluated variables.

| Variable | W | p-value | n |
|---|---:|---:|---:|
| closed_pnl | 0.1118 | 4.58e-93 | 5,000 |
| size_usd | 0.2308 | 9.68e-90 | 5,000 |
| sentiment_score | 0.8535 | 7.46e-56 | 5,000 |

Decision: reject normality in all three cases.

Interpretation: the sample is strongly non-normal, so non-parametric comparisons are better suited for sentiment-group inference.

## Variance

Levene's test for closed PnL across sentiment groups returned:

- Statistic: 20.9869
- p-value: 2.66e-17

Decision: reject equal variances.

Interpretation: sentiment groups do not have comparable PnL variance, which further supports the use of non-parametric group testing.

## Correlation

Pearson and Spearman correlations were used because Pearson measures linear association while Spearman captures monotonic relationships and is more robust to skewed financial data.

| Pair | Pearson / Spearman | Coefficient | p-value |
|---|---|---:|---:|
| closed_pnl vs size_usd | Pearson | 0.1402 | 0.0 |
| closed_pnl vs size_usd | Spearman | 0.1121 | 4.54e-220 |
| closed_pnl vs sentiment_score | Pearson | 0.0119 | 0.0245 |
| closed_pnl vs sentiment_score | Spearman | 0.1214 | 9.66e-118 |
| size_usd vs sentiment_score | Pearson | -0.0253 | 1.66e-06 |
| size_usd vs sentiment_score | Spearman | -0.0641 | 5.30e-34 |

Interpretation: the statistically significant results are weak in magnitude. Trade size has the clearest positive relationship with PnL, while sentiment score shows only a small association with profitability.

## Group Comparisons

Kruskal-Wallis testing found a statistically significant difference in closed PnL across sentiment groups.

- Statistic: 677.2053
- p-value: 3.00e-145

Decision: reject the null hypothesis of equal sentiment-group distributions.

Sentiment-group summary:

| Sentiment | Trade Count | Mean PnL | Median PnL | Win Rate |
|---|---:|---:|---:|---:|
| Extreme Greed | 5,621 | 205.82 | 0.96 | 55.33% |
| Fear | 13,869 | 128.29 | 0.00 | 38.18% |
| Greed | 11,292 | 53.99 | 0.00 | 43.57% |
| Neutral | 2,756 | 27.09 | 0.00 | 49.49% |
| Extreme Fear | 2,326 | 1.89 | 0.00 | 29.28% |

Post-hoc Mann-Whitney tests with Holm correction showed significant pairwise differences for every combination except Greed vs Neutral.

Interpretation: profitability differs materially across sentiment regimes, but not every pair is distinct after multiple-comparison correction.

## Exploratory Modeling

A Random Forest Regressor was trained with a train/test split and `random_state=42` using `size_usd`, `sentiment_score`, and `trade_hour`.

| Metric | Value |
|---|---:|
| Rows used | 35,864 |
| MAE | 178.62 |
| RMSE | 1,017.19 |
| R² | 0.2424 |

Feature importance:

| Feature | Importance |
|---|---:|
| size_usd | 0.6790 |
| trade_hour | 0.2434 |
| sentiment_score | 0.0776 |

Interpretation: this model captures some signal, but performance is modest and not strong enough for production trading use.

# Business Insights

- Extreme Greed is associated with the best observed profit profile in the matched sentiment sample.
- Extreme Fear is associated with the weakest observed profit profile and the lowest win rate.
- Trade size has the most consistent relationship with PnL among the numeric variables tested.
- Sentiment appears to matter, but the effect size is small and should be treated as contextual rather than deterministic.
- Greed and Neutral are not statistically different after Holm correction, so they should not be over-separated in interpretation.

# Recommendations

- Use sentiment as a risk-context signal, not as a standalone trading rule.
- Treat large position sizes with care because they are associated with larger PnL outcomes and larger capital exposure.
- Separate extreme sentiment regimes from mid-range regimes when reviewing performance, since they show the clearest differences.
- Avoid over-interpreting weak correlations as actionable alpha.
- Use the predictive model only as a baseline for further experimentation.

# Limitations

- The data is historical and observational, so it cannot establish causation.
- Only a subset of trades had matched sentiment scores.
- No macroeconomic, funding-rate, order-book, or liquidity variables were included.
- Leverage was not usable as a numeric feature in the cleaned dataset for the final inferential section.
- The model is exploratory rather than production-grade.
- Survivorship and selection bias may still affect the source data.

# Future Work

- Add leverage, funding-rate, liquidation, and order-book data if available.
- Test walk-forward validation and regime-aware modeling.
- Explore trader clustering once richer behavioral features exist.
- Compare results across different market windows to check stability.

# Conclusion

The project shows that Bitcoin sentiment and Hyperliquid trader outcomes are related, but only in a limited and non-linear way. Sentiment regime matters for profitability, but the most reliable signals in this dataset remain descriptive rather than predictive. The results are suitable for research and interview presentation, but not for claims of tradable edge.

# References

- Hyperliquid trader dataset
- Bitcoin Fear & Greed Index dataset
- scipy statistical tests
- scikit-learn exploratory model