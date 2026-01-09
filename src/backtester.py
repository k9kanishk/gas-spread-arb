"""Utilities for backtesting spread trading strategies."""

from __future__ import annotations

import pandas as pd


def backtest_spread(
    spread: pd.Series,
    signal: pd.Series,
    cost_per_turnover: float = 0.0,
    per_leg_cost: float = 0.0,
) -> pd.DataFrame:
    """
    Backtest a simple spread strategy.

    The strategy takes positions based on the provided trading signal and
    evaluates performance on the given spread time series.

    Parameters
    ----------
    spread : pd.Series
        Time series of spread levels (e.g., TTF minus NBP).
    signal : pd.Series
        Desired position at each time (0 for flat, +1 for long, -1 for short).
    cost_per_turnover : float, default 0.0
        Absolute transaction cost applied per unit of turnover.
    per_leg_cost : float, default 0.0
        Optional per-leg cost applied to turnover (two legs per spread).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing spread, signal, realized position, gross and net
        PnL components, and the equity curve. The columns include:
        ``spread``, ``signal``, ``position``, ``dS``, ``gross_pnl``, ``costs``,
        ``pnl``, and ``equity``.
    """

    s = spread.dropna().astype(float)
    sig = signal.reindex(s.index).fillna(0.0).astype(float)

    position = sig.shift(1).fillna(0.0)
    d_spread = s.diff().fillna(0.0)
    gross_pnl = position * d_spread

    turnover = position.diff().abs().fillna(0.0)
    costs = float(cost_per_turnover) * turnover
    if per_leg_cost > 0:
        costs += 2.0 * float(per_leg_cost) * turnover

    pnl = gross_pnl - costs
    equity = pnl.cumsum()

    return pd.DataFrame(
        {
            "spread": s,
            "signal": sig,
            "position": position,
            "dS": d_spread,
            "gross_pnl": gross_pnl,
            "costs": costs,
            "pnl": pnl,
            "equity": equity,
        }
    )
