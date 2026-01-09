# Cross-Market Gas Spread Statistical Arbitrage (TTF–NBP–JKM)

An interactive **Streamlit dashboard** that models and backtests **relative-value (spread) trades** across major natural gas benchmarks:

- **TTF** (Europe, EUR/MWh)
- **NBP** (UK, converted to EUR/MWh)
- **JKM** (Asia LNG, converted to EUR/MWh via FX and unit conversions)

The project focuses on how gas traders think: **spreads and netbacks**, not outright price forecasts.

---

## What this app does

### 1) Data alignment and normalization
The app loads daily benchmark histories from `data/raw/` and aligns them on a common date index.

It converts all hubs into a common unit and currency (EUR/MWh):

- TTF: already EUR/MWh (assumed)
- NBP: typically pence/therm → GBP/MWh → EUR/MWh (using FX)
- JKM: typically USD/MMBtu → EUR/MWh (using FX + energy unit conversion)

FX inputs (static CSV):
- `EURUSD` = USD per 1 EUR
- `GBPUSD` = USD per 1 GBP

### 2) Spread construction
Two main spreads are analyzed:

**(A) TTF–NBP hub spread**
\[
S_{TTF,NBP}(t) = TTF_{EUR/MWh}(t) - NBP_{EUR/MWh}(t)
\]

**(B) TTF–JKM netback spread**
A simplified cross-basin proxy:
\[
S_{TTF,JKM}(t) = JKM_{EUR/MWh}(t) - \text{Shipping}(t) - TTF_{EUR/MWh}(t)
\]

> Note: shipping is implemented as a user-controlled constant in EUR/MWh (a proxy).

### 3) Mean reversion modeling
Each spread is modeled as a mean-reverting process via an **AR(1)** approximation:

\[
x_t = c + \phi x_{t-1} + \varepsilon_t
\]

The dashboard reports:
- φ (persistence)
- long-run mean μ
- residual volatility (σₑ)
- **stationary volatility** (σₓ) used for z-scores
- half-life (days)

### 4) Signal generation (z-score rules)
A simple threshold strategy:

- If z-score > +entry_z → **short spread** (expect convergence)
- If z-score < -entry_z → **long spread**
- Exit when |z| < exit_z

### 5) Backtesting + performance reporting
The strategy is backtested with:
- transaction cost per turnover
- optional per-leg costs

Metrics include:
- Sharpe ratio
- max drawdown
- total PnL
- trade count
- win ratio
- holding period statistics

### 6) Parameter sweep
The dashboard runs a small grid search over (entry_z, exit_z) pairs to show how performance varies with thresholds.

---

## App features

- Interactive controls (spread selection, entry/exit z-scores, shipping cost, costs)
- Model modes:
  - **Train/Test (fixed params)**: fit on training period, trade on test period
  - **Rolling (walk-forward)**: re-estimate parameters using a rolling lookback window
- Charts:
  - spread time series
  - z-score + entry/exit bands
  - equity curve
- Tables:
  - AR(1) parameter summary
  - performance summary
  - holding period distribution
  - parameter sweep results

---

## Repository structure

```text
gas-spread-arb/
  app.py
  requirements.txt
  README.md

  data/
    raw/
      ttf_prices.csv
      nbp_prices.csv
      jkm_prices.csv
      fx_rates.csv
    processed/
      clean_prices.csv
      normalized_prices.csv
      spreads.csv

  src/
    data_loader.py
    spreads.py
    models.py
    signals.py
    backtester.py
    metrics.py
```

Data requirements (important)
Place these CSVs in data/raw/:

ttf_prices.csv
Must contain:

Date column

a price column (ideally named Price)

nbp_prices.csv
Must contain:

Date

price column (often pence/therm)

jkm_prices.csv
Must contain:

Date

price column (often USD/MMBtu)

fx_rates.csv
Must contain:

Date

EURUSD (USD per 1 EUR)

GBPUSD (USD per 1 GBP)

Missing FX values on holidays/weekends are fine; the pipeline forward-fills where needed.

How to run locally
```bash
# 1) Create environment (optional)
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run Streamlit
streamlit run app.py
```

Interpretation notes (read before judging results)
This is an educational/stat-arb style prototype intended for portfolio demonstration.

What the backtest IS:

A clean, reproducible spread analytics + signal + backtest pipeline

A demonstration of cross-market normalization (FX, unit conversions)

A relative-value mindset applied to gas hubs and LNG netbacks

What the backtest is NOT:

A fully tradeable implementation of real hub/LNG strategies

A substitute for actual physical constraints (capacity, nominations, credit, margining)

A full futures contract roll / liquidity / slippage / bid-ask model

A robust “production strategy” claim

Key limitations and assumptions
Constant shipping proxy (user slider) is a simplification.

Constant shipping shifts spread level but does not change Δspread-based PnL materially.

Data source quality depends on the CSVs provided (often scraped/free sources).

No contract roll logic (e.g., front-month futures) unless embedded in input data.

Costs are simplified: flat turnover-based costs and optional per-leg cost.

Suggested extensions (future work)
Replace constant shipping with a time-varying freight proxy or LNG shipping index.

Add structural breaks/regime tests (e.g., pre/post 2021 energy crisis).

Add volatility targeting position sizing (risk parity on spread volatility).

Add realistic tradable legs (futures curves, roll schedules).

Add cross-validation / more robust walk-forward evaluation.

Interview-ready summary
“I built a cross-market spread analytics dashboard for TTF–NBP and TTF–JKM netbacks.
I normalized prices into a common currency/unit using FX and energy conversions, modeled each spread as a mean-reverting AR(1) process, generated z-score convergence signals, and evaluated performance under train/test and rolling walk-forward regimes with transaction costs.”

Disclaimer
This project is for educational and research purposes only and does not constitute investment advice.

If you want, I can also write a **short “Project write-up”** section for your portfolio website (3–6 bullets + 1 paragraph) that matches how recruiters skim.
