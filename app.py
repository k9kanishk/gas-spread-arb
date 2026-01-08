"""Streamlit dashboard for gas spread trading signals and backtests."""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yfinance as yf

from src.backtester import backtest_spread
from src.data_loader import load_raw_prices, save_clean_prices
from src.metrics import summarize_backtest
from src.models import fit_ar1, half_life
from src.signals import generate_signal
from src.spreads import build_spreads, normalize_to_eur_mwh

RAW_DIR = Path("data") / "raw"
PROCESSED_DIR = Path("data") / "processed"


def ensure_fx_data(start_date: str = "2015-01-01") -> Path:
    """Ensure data/raw/fx_rates.csv exists; download if needed."""

    fx_path = RAW_DIR / "fx_rates.csv"
    if fx_path.exists():
        return fx_path

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    tickers = ["EURUSD=X", "GBPUSD=X"]
    fx = yf.download(tickers, start=start_date)["Adj Close"]
    fx.columns = ["EURUSD", "GBPUSD"]
    fx.index.name = "Date"
    fx.to_csv(fx_path)

    return fx_path


@st.cache_data(show_spinner=False)
def load_pipeline(shipping_cost: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full data pipeline used by the app."""

    ensure_fx_data()

    raw_df = load_raw_prices()
    save_clean_prices(raw_df)

    normalized = normalize_to_eur_mwh(raw_df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(PROCESSED_DIR / "normalized_prices.csv")

    spreads = build_spreads(normalized, shipping_cost=shipping_cost)
    spreads.to_csv(PROCESSED_DIR / "spreads.csv")
    return raw_df, normalized, spreads


def format_parameters(params: dict) -> pd.DataFrame:
    """Convert AR(1) parameters to a display-friendly dataframe."""

    return pd.DataFrame(
        {
            "Value": {
                "Constant": params["const"],
                "Phi": params["phi"],
                "Long-run mean": params["mu"],
                "Residual sigma": params["sigma"],
                "Half-life (days)": half_life(params["phi"]),
            }
        }
    )


st.set_page_config(page_title="Gas Spread Strategy Dashboard", layout="wide")
st.title("Gas Spread Statistical Arbitrage Dashboard")
st.markdown(
    """
    Explore spread relationships across TTF, NBP, and JKM, fit mean-reversion models,
    and evaluate simple z-score based trading rules. Adjust thresholds and costs in
    the sidebar to see how they affect signal generation and performance.
    """
)

with st.sidebar:
    st.header("Controls")
    selected_spread = st.selectbox("Spread", options=["TTF_NBP", "TTF_JKM_netback"], index=0)
    entry_z = st.slider("Entry z-score", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    exit_z = st.slider("Exit z-score", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    shipping_cost = st.slider(
        "Shipping cost (EUR/MWh) for TTF vs JKM netback",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1,
        help="Used only when the selected spread is TTF_JKM_netback",
    )
    transaction_cost = st.slider(
        "Transaction cost per turnover",
        min_value=0.0,
        max_value=1.0,
        value=0.02,
        step=0.01,
    )

_clean_prices, normalized_prices, spreads = load_pipeline(shipping_cost=shipping_cost)

spread_series = spreads[selected_spread].dropna()

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader(f"{selected_spread} spread")
    st.line_chart(spread_series, height=280)
with col2:
    st.subheader("AR(1) estimates")
    ar1_params = fit_ar1(spread_series)
    st.dataframe(format_parameters(ar1_params), use_container_width=True)

signal, zscores = generate_signal(
    spread=spread_series,
    mu=ar1_params["mu"],
    sigma=ar1_params["sigma"],
    entry_z=entry_z,
    exit_z=exit_z,
)

zscore_df = pd.DataFrame({"zscore": zscores}).dropna()

z_line = (
    alt.Chart(zscore_df.reset_index().rename(columns={"index": "Date"}))
    .mark_line()
    .encode(x="Date:T", y="zscore:Q")
)
thresholds = []
for level, color in [
    (entry_z, "#d62728"),
    (-entry_z, "#d62728"),
    (exit_z, "#2ca02c"),
    (-exit_z, "#2ca02c"),
]:
    thresholds.append(
        alt.Chart(pd.DataFrame({"z": [level]}))
        .mark_rule(strokeDash=[4, 4], color=color)
        .encode(y="z:Q")
    )

st.subheader("Z-score and thresholds")
st.altair_chart(alt.layer(z_line, *thresholds).properties(height=300), use_container_width=True)

backtest = backtest_spread(spread_series, signal, cost_per_turnover=float(transaction_cost))

st.subheader("Equity curve")
st.line_chart(backtest["equity"], height=280)

st.subheader("Performance summary")
metrics_table = summarize_backtest(backtest, name=selected_spread)
st.dataframe(metrics_table.to_frame("Value"), use_container_width=True)

st.caption(
    "Data is prepared from raw CSVs in `data/raw` (with FX downloaded if missing) and cached for fast reloads. "
    "Adjust the sidebar to explore different thresholds and costs."
)
