"""Utilities for loading and persisting price data.

The functions in this module assume the following raw CSV files exist under
``data/raw``:

- ``ttf_prices.csv``: date column named ``Date`` and a single price column for TTF in EUR/MWh
- ``nbp_prices.csv``: date column named ``Date`` and a price column in pence/therm
- ``jkm_prices.csv``: date column named ``Date`` and a price column in USD/MMBtu
- ``fx_rates.csv``: date column named ``Date`` and FX columns ``EURUSD`` and ``GBPUSD`` (both quoted as USD per unit)

If any price column is not explicitly named, the first non-date column will be
used as the price series. All series are aligned on a common daily index, then
forward-filled and truncated to remove any leading missing values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

RAW_DATA_DIR = Path("data") / "raw"
PROCESSED_DATA_DIR = Path("data") / "processed"


def _load_price_series(path: Path, target_name: str, expected_price_column: str = "Price") -> pd.Series:
    """Load a single price series from ``path`` and return it as a ``pd.Series``.

    The CSV is read with ``parse_dates=True`` and ``index_col='Date'``. If the
    expected price column is absent, the first non-date column is used instead.

    Args:
        path: Location of the CSV file.
        target_name: Desired column name in the returned series.
        expected_price_column: Name of the price column to look for.

    Returns:
        ``pd.Series`` indexed by ``Date`` with the requested ``target_name``.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If no suitable price column can be identified.
    """

    if not path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {path}")

    df = pd.read_csv(path, parse_dates=True, index_col="Date")

    price_column = expected_price_column if expected_price_column in df.columns else None
    if price_column is None:
        candidates: Iterable[str] = (col for col in df.columns if col.lower() != "date")
        try:
            price_column = next(iter(candidates))
        except StopIteration as exc:  # pragma: no cover - defensive branch
            raise ValueError(
                f"No price column found in {path}. Expected '{expected_price_column}' or any non-date column"
            ) from exc

    series = df[price_column].rename(target_name)
    return series.sort_index()


def load_raw_prices() -> pd.DataFrame:
    """Load and align raw price and FX series.

    Returns a dataframe with columns ``[TTF, NBP, JKM, EURUSD, GBPUSD]`` aligned
    on a common daily ``Date`` index. Missing values are forward-filled and any
    leading gaps are removed.

    Returns:
        A ``pd.DataFrame`` containing the aligned series.
    """

    ttf = _load_price_series(RAW_DATA_DIR / "ttf_prices.csv", target_name="TTF")
    nbp = _load_price_series(RAW_DATA_DIR / "nbp_prices.csv", target_name="NBP")
    jkm = _load_price_series(RAW_DATA_DIR / "jkm_prices.csv", target_name="JKM")

    fx_path = RAW_DATA_DIR / "fx_rates.csv"
    if not fx_path.exists():
        raise FileNotFoundError(f"Expected FX CSV not found: {fx_path}")
    fx = pd.read_csv(fx_path, parse_dates=True, index_col="Date")
    if not {"EURUSD", "GBPUSD"}.issubset(fx.columns):
        raise ValueError("fx_rates.csv must contain 'EURUSD' and 'GBPUSD' columns")
    fx = fx[["EURUSD", "GBPUSD"]].sort_index()

    combined = pd.concat([ttf, nbp, jkm, fx], axis=1, join="outer").sort_index()
    filled = combined.ffill()
    cleaned = filled.dropna(how="any")
    return cleaned


def save_clean_prices(df: pd.DataFrame) -> Path:
    """Persist the cleaned dataset to ``data/processed/clean_prices.csv``.

    Args:
        df: Dataframe containing price and FX series, typically from
            :func:`load_raw_prices`.

    Returns:
        Path to the saved CSV file.
    """

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "clean_prices.csv"
    df.sort_index().to_csv(output_path)
    return output_path


__all__ = ["load_raw_prices", "save_clean_prices"]
