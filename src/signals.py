"""Signal generation utilities based on spread z-scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(spread: pd.Series, mu: float, sigma: float) -> pd.Series:
    """Compute the z-score of a spread series.

    Args:
        spread: Spread series.
        mu: Long-run mean of the spread.
        sigma: Standard deviation of the spread.

    Returns:
        A :class:`pandas.Series` of z-scores aligned to the original index. If
        ``sigma`` is zero or NaN, returns all-NaN values.
    """

    if sigma == 0 or np.isnan(sigma):
        return pd.Series(np.nan, index=spread.index)

    return (spread - mu) / sigma


def generate_signal(
    spread: pd.Series,
    mu: float,
    sigma: float,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> tuple[pd.Series, pd.Series]:
    """Generate trading signals based on z-score thresholds.

    Args:
        spread: Spread series.
        mu: Long-run mean of the spread.
        sigma: Standard deviation of the spread.
        entry_z: Z-score threshold to open a position.
        exit_z: Z-score threshold to close an existing position.

    Returns:
        A tuple ``(signal_series, zscore_series)`` where ``signal_series``
        contains position signals (``-1`` short, ``0`` flat, ``+1`` long).
    """

    zscores = zscore(spread, mu, sigma)
    positions: list[int] = []
    current_pos = 0

    for z in zscores:
        if np.isnan(z):
            positions.append(current_pos)
            continue

        if current_pos == 0:
            if z > entry_z:
                current_pos = -1
            elif z < -entry_z:
                current_pos = 1
        else:
            if abs(z) < exit_z:
                current_pos = 0

        positions.append(current_pos)

    signal_series = pd.Series(positions, index=spread.index)
    return signal_series, zscores
