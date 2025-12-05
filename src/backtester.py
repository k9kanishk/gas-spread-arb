"""Utilities for backtesting spread trading strategies."""

from __future__ import annotations

import pandas as pd


def backtest_spread(
    spread: pd.Series,
    signal: pd.Series,
    cost_per_turnover: float = 0.0,
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
        Absolute transaction cost applied whenever the position changes.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing spread, signal, realized position, gross and net
        PnL components, and the equity curve. The columns include:
        ``spread``, ``signal``, ``position``, ``gross_pnl``, ``costs``,
        ``net_pnl``, and ``equity``.
    """

    # Align inputs to ensure consistent indices.
    spread_aligned, signal_aligned = spread.align(signal, join="inner")

    d_spread = spread_aligned.diff().fillna(0.0)

    # Yesterday's signal determines today's position.
    position = signal_aligned.shift(1).fillna(0.0)

    gross_pnl = position * d_spread

    signal_change = signal_aligned.diff().fillna(signal_aligned)
    turnover_events = signal_change.ne(0)
    costs = turnover_events.astype(float) * float(cost_per_turnover)

    net_pnl = gross_pnl - costs
    equity = net_pnl.cumsum()

    return pd.DataFrame(
        {
            "spread": spread_aligned,
            "signal": signal_aligned,
            "position": position,
            "gross_pnl": gross_pnl,
            "costs": costs,
            "net_pnl": net_pnl,
            "equity": equity,
        }
    )
