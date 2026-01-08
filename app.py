"""Streamlit dashboard for gas spread trading signals and backtests."""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.backtester import backtest_spread
from src.data_loader import load_raw_prices, save_clean_prices
from src.metrics import summarize_backtest
from src.models import fit_ar1, half_life, rolling_ar1_params
from src.signals import generate_signal, signal_from_zscores
from src.spreads import build_spreads, normalize_to_eur_mwh

PROCESSED_DIR = Path("data") / "processed"
BASELINE_SHIPPING = 3.0


@st.cache_data(show_spinner=False)
def load_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalize raw price data for reuse across app updates."""

    try:
        raw_df = load_raw_prices()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(
            "Fix: create data/raw/fx_rates.csv with columns: Date, EURUSD, GBPUSD "
            "(USD per 1 EUR, USD per 1 GBP)."
        )
        st.stop()
    save_clean_prices(raw_df)

    normalized = normalize_to_eur_mwh(raw_df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(PROCESSED_DIR / "normalized_prices.csv")

    return raw_df, normalized


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


def holding_periods(position: pd.Series) -> pd.Series:
    """Compute holding period lengths (in days) for non-flat positions."""

    active = position[position != 0]
    if active.empty:
        return pd.Series(dtype=int)

    groups = active.ne(active.shift()).cumsum()
    return active.groupby(groups).size()


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
    refit_on_shipping = st.checkbox(
        "Refit model when shipping changes",
        value=True,
        help="Disable to lock model calibration at the baseline shipping cost.",
    )
    model_mode = st.radio(
        "Model mode",
        options=["Train/Test (fixed params)", "Rolling (walk-forward)"],
        index=0,
    )
    lookback_days = st.slider(
        "Rolling lookback (days)",
        min_value=60,
        max_value=756,
        value=252,
        step=21,
        help="Used only for rolling walk-forward mode.",
    )
    transaction_cost = st.slider(
        "Transaction cost per turnover",
        min_value=0.0,
        max_value=1.0,
        value=0.02,
        step=0.01,
    )
    per_leg_cost = st.slider(
        "Optional per-leg cost (each leg)",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.01,
        help="If set, applied per leg and added to the turnover cost.",
    )

raw_prices, normalized_prices = load_base()
spreads = build_spreads(normalized_prices, shipping_cost=shipping_cost)
spreads.to_csv(PROCESSED_DIR / "spreads.csv")

st.write("Shipping:", shipping_cost)
st.write("Last netback:", float(spreads["TTF_JKM_netback"].dropna().iloc[-1]))

spread_series = spreads[selected_spread].dropna()

if refit_on_shipping:
    calibration_spread = spread_series
else:
    baseline_spreads = build_spreads(normalized_prices, shipping_cost=BASELINE_SHIPPING)
    calibration_spread = baseline_spreads[selected_spread].dropna()

min_date = spread_series.index.min().date()
max_date = spread_series.index.max().date()

default_train_end = pd.Timestamp("2019-12-31").date()
if default_train_end < min_date or default_train_end > max_date:
    default_train_end = max_date

if model_mode == "Train/Test (fixed params)":
    train_end_date = st.sidebar.slider(
        "Training end date",
        min_value=min_date,
        max_value=max_date,
        value=default_train_end,
        format="YYYY-MM-DD",
    )
    train_end = pd.Timestamp(train_end_date)
else:
    train_end = None

cost_per_turnover = float(transaction_cost) + (2 * float(per_leg_cost))

if model_mode == "Train/Test (fixed params)":
    train = calibration_spread.loc[:train_end]
    test = spread_series.loc[train_end:]

    if train.empty or test.empty:
        st.warning("Train/test split leaves no data on one side. Adjust the date.")
        st.stop()

    ar1_params = fit_ar1(train)
    signal, zscores = generate_signal(
        spread=test,
        mu=ar1_params["mu"],
        sigma=ar1_params["sigma"],
        entry_z=entry_z,
        exit_z=exit_z,
    )
    spread_for_backtest = test
    zscores_for_plot = zscores
    params_table = format_parameters(ar1_params)
else:
    rolling_params = rolling_ar1_params(calibration_spread, lookback=lookback_days)
    if rolling_params.dropna().empty:
        st.warning("Rolling window is longer than available data. Reduce the lookback.")
        st.stop()
    mu_series = rolling_params["mu"]
    sigma_series = rolling_params["sigma"]
    zscores = (spread_series - mu_series) / sigma_series
    signal = signal_from_zscores(zscores, entry_z=entry_z, exit_z=exit_z)
    spread_for_backtest = spread_series
    zscores_for_plot = zscores
    latest_params = rolling_params.dropna().iloc[-1]
    params_table = pd.DataFrame(
        {
            "Value": {
                "Constant": latest_params["const"],
                "Phi": latest_params["phi"],
                "Long-run mean": latest_params["mu"],
                "Residual sigma": latest_params["sigma"],
                "Half-life (days)": half_life(latest_params["phi"]),
            }
        }
    )

backtest = backtest_spread(spread_for_backtest, signal, cost_per_turnover=cost_per_turnover)

summary_tab, sweep_tab = st.tabs(["Overview", "Parameter sweep"])

with summary_tab:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"{selected_spread} spread")
        st.line_chart(spread_series, height=280)
    with col2:
        st.subheader("AR(1) estimates")
        st.dataframe(params_table, use_container_width=True)

    zscore_df = pd.DataFrame({"zscore": zscores_for_plot}).dropna()

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

    st.subheader("Equity curve")
    st.line_chart(backtest["equity"], height=280)

    st.subheader("Performance summary")
    metrics_table = summarize_backtest(backtest, name=selected_spread)
    st.dataframe(metrics_table.to_frame("Value"), use_container_width=True)

    st.subheader("Holding period stats")
    durations = holding_periods(backtest["position"])
    if durations.empty:
        st.caption("No positions opened in the sample.")
    else:
        stats = pd.Series(
            {
                "average_days": durations.mean(),
                "median_days": durations.median(),
                "min_days": durations.min(),
                "max_days": durations.max(),
            }
        )
        st.dataframe(stats.to_frame("Value"), use_container_width=True)
        st.bar_chart(durations.value_counts().sort_index(), height=200)

with sweep_tab:
    st.subheader("Parameter sweep")
    entry_grid = [1.5, 2.0, 2.5, 3.0]
    exit_grid = [0.25, 0.5, 0.75]

    sweep_rows = []
    for entry in entry_grid:
        for exit_value in exit_grid:
            if model_mode == "Train/Test (fixed params)":
                signal_sweep, _ = generate_signal(
                    spread=spread_for_backtest,
                    mu=ar1_params["mu"],
                    sigma=ar1_params["sigma"],
                    entry_z=entry,
                    exit_z=exit_value,
                )
            else:
                signal_sweep = signal_from_zscores(zscores_for_plot, entry_z=entry, exit_z=exit_value)
            bt = backtest_spread(
                spread_for_backtest,
                signal_sweep,
                cost_per_turnover=cost_per_turnover,
            )
            metrics = summarize_backtest(bt, name=f"{entry}/{exit_value}")
            sweep_rows.append(
                {
                    "entry_z": entry,
                    "exit_z": exit_value,
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                    "total_pnl": metrics["total_pnl"],
                }
            )

    sweep_df = pd.DataFrame(sweep_rows).set_index(["entry_z", "exit_z"])
    st.dataframe(sweep_df, use_container_width=True)

st.caption(
    "Data is prepared from raw CSVs in `data/raw` and cached for fast reloads. "
    "Adjust the sidebar to explore different thresholds and costs."
)
