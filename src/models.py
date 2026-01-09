"""Model utilities for fitting mean-reverting processes to spread data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def half_life(phi: float) -> float:
    """Compute the mean-reversion half-life for an AR(1) process.

    Uses the relationship :math:`t_{1/2} = \ln(2) / -\ln(\phi)` when ``phi`` is
    within the stationary range ``(0, 1)``. For values outside ``(-1, 1)`` or
    non-positive coefficients, the process is non-stationary and the half-life
    is treated as infinite.

    Args:
        phi: The AR(1) coefficient.

    Returns:
        The half-life in the same time units as the input series, or ``np.inf``
        when the process is non-stationary.
    """

    if not -1 < phi < 1 or phi <= 0:
        return np.inf

    return np.log(2) / -np.log(phi)


def fit_ar1(series: pd.Series) -> dict:
    """Fit an AR(1) model ``S_t = c + phi * S_{t-1}`` to a spread series.

    Args:
        series: Time series of spread levels.

    Returns:
        A dictionary containing ``constant`` (c), ``phi``, ``mu`` (long-run mean),
        ``sigma_eps`` (residual standard deviation), ``sigma_x`` (stationary
        standard deviation), and ``half_life_days``.
    """

    cleaned = series.dropna().astype(float)
    if len(cleaned) < 2:
        raise ValueError("At least two observations are required to fit AR(1) after dropping NaNs.")

    x = cleaned.shift(1).dropna()
    y = cleaned.loc[x.index]

    x_mean, y_mean = x.mean(), y.mean()
    phi = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
    constant = y_mean - phi * x_mean

    resid = y - (constant + phi * x)
    sigma_eps = resid.std(ddof=1)

    mu = constant / (1 - phi) if abs(1 - phi) > 1e-12 else np.nan
    denom = max(1e-12, 1 - phi**2)
    sigma_x = sigma_eps / np.sqrt(denom)

    half_life_days = (-np.log(2) / np.log(abs(phi))) if 0 < abs(phi) < 1 else np.nan

    return {
        "constant": constant,
        "phi": phi,
        "mu": mu,
        "sigma_eps": float(sigma_eps),
        "sigma_x": float(sigma_x),
        "half_life_days": float(half_life_days),
    }


def rolling_ar1_params(spread: pd.Series, lookback: int = 252) -> pd.DataFrame:
    """Compute rolling AR(1) parameters using a fixed lookback window.

    Args:
        spread: Time series of spread levels.
        lookback: Number of historical observations to use for each fit.

    Returns:
        DataFrame with columns ``constant``, ``phi``, ``mu``, ``sigma_eps``, and
        ``sigma_x`` aligned to the input index. Values are NaN until enough
        history is available.
    """

    cleaned = spread.dropna()
    params = pd.DataFrame(
        index=cleaned.index,
        columns=["constant", "phi", "mu", "sigma_eps", "sigma_x"],
        dtype=float,
    )

    values = cleaned.to_numpy()
    for idx in range(lookback, len(cleaned)):
        window = values[idx - lookback : idx]
        if len(window) < 2:
            continue

        x = window[:-1]
        y = window[1:]
        var_x = np.var(x, ddof=0)
        if var_x == 0:
            continue

        cov_xy = np.mean((x - x.mean()) * (y - y.mean()))
        phi = cov_xy / var_x
        const = float(y.mean() - phi * x.mean())
        mu = const / (1 - phi) if phi != 1 else np.nan
        residuals = y - (const + phi * x)
        sigma_eps = float(np.std(residuals, ddof=1))
        denom = max(1e-12, 1 - phi**2)
        sigma_x = sigma_eps / np.sqrt(denom)

        params.iloc[idx] = [const, phi, mu, sigma_eps, sigma_x]

    return params.reindex(spread.index)
