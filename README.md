# Cross-Market Gas Spread & Arbitrage Model (TTF–NBP–JKM)

## Overview
This project explores cross-basin natural gas arbitrage opportunities by modeling the price relationships among the Dutch TTF hub, UK NBP hub, and Asian JKM benchmark. The strategy focuses on spread behavior, mean reversion tendencies, and z-score-based signals to enter and exit trades across the interconnected gas markets.

## Data Inputs and Normalization
- **TTF**: Quoted in **EUR/MWh**.
- **NBP**: Quoted in **pence/therm**.
- **JKM**: Quoted in **USD/MMBtu**.
- **FX series**: Currency conversions required to express all series in **EUR/MWh**.

Normalization steps:
1. Convert NBP pence/therm to GBP/therm, then apply GBP→EUR FX and therm→MMBtu→MWh energy unit conversions to reach EUR/MWh.
2. Convert JKM USD/MMBtu to EUR/MMBtu via USD→EUR FX, then MM Btu→MWh to reach EUR/MWh.
3. Align calendars and, if necessary, interpolate missing FX values to maintain consistent spread calculations.

## Modeling Approach
- **Spreads**: Primary focus on pairwise spreads (e.g., TTF–NBP, TTF–JKM) expressed in EUR/MWh.
- **AR(1) dynamics**: Each spread is fit with an AR(1) process to estimate drift toward a **long-run mean** and compute **half-life** of mean reversion.
- **Signal rules**: Entry and exit are driven by **z-scores** of the de-meaned spread relative to its rolling standard deviation. Example: enter when |z| exceeds a threshold; exit when |z| collapses below a tighter band or crosses zero.

## Backtest Mechanics
- **Positioning**: Simulate 1 unit of the selected spread (long/short the legs implied by the spread sign).
- **PnL**: Daily mark-to-market on the spread, incorporating currency and unit-consistent prices.
- **Transaction costs**: Deduct per-leg costs on trade entry/exit to reflect realistic slippage/fees.
- **Metrics**: Track total PnL, hit ratio, average trade PnL, max drawdown, and Sharpe/Sortino where applicable.

## Project Structure
- `data/raw/`: Place source CSVs here (TTF, NBP, JKM, and FX). The notebooks and app expect these filenames to be discoverable in this folder.
- `notebooks/`: Exploratory analysis, normalization steps, AR(1) estimation, and backtesting walkthroughs.
- `src/`: Core utilities for loading data, transforming units, computing spreads, and generating signals.
- `docs/`: Additional background materials and references.

## Setup and Usage
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Data**: Add raw CSV files to `data/raw/` with price series and FX rates; ensure date columns align across sources.
3. **Run notebooks**: Launch Jupyter (`jupyter notebook` or `jupyter lab`) and open the notebooks in `notebooks/` to reproduce normalization, AR(1) fitting, and backtest analysis.
4. **Streamlit app**: Start the UI to explore spreads and signals interactively:
   ```bash
   streamlit run src/app.py
   ```

## Strategy Talking Points (Interviews)
- Motivated by cross-market price linkage between European hubs and Asian LNG benchmarks; highlights basis convergence/divergence dynamics.
- Combines financial engineering (currency/unit normalization) with time-series modeling (AR(1), half-life) and systematic trading rules (z-score thresholds).
- Shows full research loop: data cleaning, model estimation, parameter interpretation (mean/half-life), backtesting with costs, and visualization via notebooks/Streamlit.

