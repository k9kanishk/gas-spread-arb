"""Performance metrics for spread backtests."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def sharpe(pnl: pd.Series, periods_per_year: int = 252) -> float:
    """Compute the annualized Sharpe ratio for a PnL series.

    Parameters
    ----------
    pnl : pd.Series
        Series of periodic profit and loss values.
    periods_per_year : int, default 252
        Number of periods in a calendar year (e.g., 252 for daily data).

    Returns
    -------
    float
        The annualized Sharpe ratio. Returns ``0.0`` when volatility is zero or
        the input series is empty.
    """

    pnl = pnl.dropna()
    if pnl.empty:
        return 0.0

    mean = pnl.mean()
    vol = pnl.std(ddof=0)
    if vol == 0:
        return 0.0

    return float((mean / vol) * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """
    Calculate the maximum drawdown of an equity curve.

    Parameters
    ----------
    equity : pd.Series
        Equity curve values.

    Returns
    -------
    float
        The maximum drawdown (most negative peak-to-trough move). Returns
        ``0.0`` for empty input.
    """

    equity = equity.dropna()
    if equity.empty:
        return 0.0

    rolling_peak = equity.cummax()
    drawdown = equity - rolling_peak
    return float(drawdown.min())


def summarize_backtest(bt: pd.DataFrame, name: str) -> pd.Series:
    """
    Summarize key performance metrics from a backtest result.

    Parameters
    ----------
    bt : pd.DataFrame
        Backtest output, expected to contain ``signal``, ``net_pnl``, and
        ``equity`` columns.
    name : str
        Name for the strategy or spread.

    Returns
    -------
    pd.Series
        Summary statistics including Sharpe, max drawdown, total PnL, trade
        count, and win ratio.
    """

    signal = bt["signal"]
    net_pnl = bt["net_pnl"]
    equity = bt["equity"]

    summary = pd.Series(dtype=float, name=name)
    summary.loc["sharpe"] = sharpe(net_pnl)
    summary.loc["max_drawdown"] = max_drawdown(equity)
    summary.loc["total_pnl"] = float(net_pnl.sum())

    signal_change = signal.diff().fillna(signal)
    summary.loc["trades"] = float(signal_change.ne(0).sum())

    non_zero_pnl = net_pnl[net_pnl != 0]
    if non_zero_pnl.empty:
        win_ratio: float | np.floating[Any] = float("nan")
    else:
        win_ratio = (non_zero_pnl > 0).mean()
    summary.loc["win_ratio"] = float(win_ratio)

    return summary
