"""Model utilities for fitting mean-reverting processes to spread data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


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


def fit_ar1(spread: pd.Series) -> dict:
    """Fit an AR(1) model ``S_t = c + phi * S_{t-1}`` to a spread series.

    The function removes missing values, constructs the lagged regressor, and
    estimates parameters via ordinary least squares using ``statsmodels``.

    Args:
        spread: Time series of spread levels.

    Returns:
        A dictionary containing ``const`` (c), ``phi``, ``mu`` (long-run mean),
        ``sigma`` (standard deviation of residuals), and the fitted ``model``
        object.

    Raises:
        ValueError: If fewer than two observations remain after dropping NaNs.
    """

    cleaned = spread.dropna()
    if len(cleaned) < 2:
        raise ValueError("At least two observations are required to fit AR(1) after dropping NaNs.")

    y = cleaned.iloc[1:]
    x = sm.add_constant(cleaned.shift(1).iloc[1:])

    model = sm.OLS(y, x).fit()

    const = model.params.get("const", model.params.iloc[0])
    phi = model.params.drop("const", errors="ignore").iloc[0]

    mu = const / (1 - phi) if phi != 1 else np.inf
    sigma = float(np.sqrt(model.mse_resid))

    return {
        "const": const,
        "phi": phi,
        "mu": mu,
        "sigma": sigma,
        "model": model,
    }


def rolling_ar1_params(spread: pd.Series, lookback: int = 252) -> pd.DataFrame:
    """Compute rolling AR(1) parameters using a fixed lookback window.

    Args:
        spread: Time series of spread levels.
        lookback: Number of historical observations to use for each fit.

    Returns:
        DataFrame with columns ``const``, ``phi``, ``mu``, and ``sigma`` aligned
        to the input index. Values are NaN until enough history is available.
    """

    cleaned = spread.dropna()
    params = pd.DataFrame(index=cleaned.index, columns=["const", "phi", "mu", "sigma"], dtype=float)

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
        sigma = float(np.std(residuals, ddof=0))

        params.iloc[idx] = [const, phi, mu, sigma]

    return params.reindex(spread.index)
