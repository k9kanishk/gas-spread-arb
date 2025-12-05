# Project Structure Proposal: Cross-Market Gas Spread & Arbitrage Model

## Tree-style layout
```
.
├── README.md
├── data
│   └── raw
│       ├── ttf.csv
│       ├── nbp.csv
│       └── jkm.csv
├── docs
│   └── project_structure.md
├── notebooks
│   └── exploration.ipynb
├── src
│   ├── __init__.py
│   ├── config
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   └── preprocess.py
│   ├── features
│   │   ├── __init__.py
│   │   └── spreads.py
│   ├── models
│   │   ├── __init__.py
│   │   └── mean_reversion.py
│   ├── backtest
│   │   ├── __init__.py
│   │   ├── signals.py
│   │   ├── simulation.py
│   │   └── metrics.py
│   ├── visualization
│   │   ├── __init__.py
│   │   └── plots.py
│   ├── app.py
│   └── cli.py
├── requirements.txt
└── tests
    ├── __init__.py
    ├── test_spreads.py
    ├── test_mean_reversion.py
    ├── test_backtest.py
    └── test_app.py
```

## Module responsibilities
- `src/config/settings.py`: Centralized configuration defaults (paths, model parameters, default thresholds, default shipping costs, transaction cost assumptions).
- `src/data/loaders.py`: Load raw CSV gas prices and fetch FX rates from `yfinance`, enforce date parsing, and align daily frequency.
- `src/data/preprocess.py`: Clean/normalize raw series, convert units to EUR/MWh using FX rates and any unit conversions.
- `src/features/spreads.py`: Construct spreads (TTF–NBP, TTF–JKM netback), including FX conversions and shipping cost adjustments.
- `src/models/mean_reversion.py`: Fit AR(1) mean-reverting model on spread levels; provide z-score calculation and parameter summaries.
- `src/backtest/signals.py`: Generate entry/exit trading signals based on z-score thresholds and direction rules.
- `src/backtest/simulation.py`: Run backtests for 1-unit spread trades, track positions, PnL, equity curve, and trade logs.
- `src/backtest/metrics.py`: Compute performance stats (Sharpe, max drawdown, total PnL, number of trades, win ratio) from backtest output.
- `src/visualization/plots.py`: Matplotlib plotting helpers for spreads, z-scores, and equity curves to reuse across CLI/Streamlit.
- `src/app.py`: Streamlit dashboard to pick spread, adjust thresholds/shipping/transaction costs, and visualize charts and metrics.
- `src/cli.py`: Command-line interface to run backtests, optionally export results.
- `notebooks/exploration.ipynb`: Ad-hoc EDA for understanding series and verifying conversions.
- `tests/`: Unit tests for spreads construction, mean-reversion fitting, backtest simulation, and Streamlit app utilities.

## Expected raw CSV files in `data/raw/`
- `ttf.csv`: Daily TTF prices (e.g., front-month) with `Date` and `Price` columns.
- `nbp.csv`: Daily NBP prices with `Date` and `Price` columns.
- `jkm.csv`: Daily JKM prices with `Date` and `Price` columns.

## Suggested `requirements.txt`
```
pandas
numpy
statsmodels
matplotlib
streamlit
scipy
yfinance
```
