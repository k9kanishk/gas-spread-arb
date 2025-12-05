"""Spread construction and unit normalization helpers."""

from __future__ import annotations

import pandas as pd

THERM_TO_MWH = 0.0293071
MMBTU_TO_MWH = 0.293071


def normalize_to_eur_mwh(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TTF, NBP, and JKM prices to EUR/MWh.

    The input dataframe must contain columns ``TTF``, ``NBP``, ``JKM``,
    ``EURUSD``, and ``GBPUSD``. The returned dataframe keeps the same index and
    exposes columns ``TTF_EUR_MWh``, ``NBP_EUR_MWh``, and ``JKM_EUR_MWh`` in
    consistent units.

    Conversion logic:
    - ``TTF`` is assumed already quoted in EUR/MWh.
    - ``NBP`` is quoted in pence/therm. It is converted to GBP/therm, then
      GBP/MWh using ``THERM_TO_MWH``, and finally to EUR/MWh using the implied
      EUR/GBP rate ``GBPUSD / EURUSD``.
    - ``JKM`` is quoted in USD/MMBtu. It is converted to EUR/MMBtu using
      ``EURUSD`` and then to EUR/MWh using ``MMBTU_TO_MWH``.

    Args:
        df: DataFrame with raw price and FX columns.

    Returns:
        A new ``pd.DataFrame`` with normalized EUR/MWh prices.
    """

    required = {"TTF", "NBP", "JKM", "EURUSD", "GBPUSD"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    norm = pd.DataFrame(index=df.index)
    norm["TTF_EUR_MWh"] = df["TTF"].astype(float)

    nbp_pence = df["NBP"].astype(float)
    nbp_gbp_per_therm = nbp_pence / 100.0
    nbp_gbp_per_mwh = nbp_gbp_per_therm / THERM_TO_MWH
    gbp_to_eur = df["GBPUSD"] / df["EURUSD"]
    norm["NBP_EUR_MWh"] = nbp_gbp_per_mwh * gbp_to_eur

    jkm_usd = df["JKM"].astype(float)
    jkm_eur_per_mmbtu = jkm_usd / df["EURUSD"]
    norm["JKM_EUR_MWh"] = jkm_eur_per_mmbtu / MMBTU_TO_MWH

    return norm


def build_spreads(norm_df: pd.DataFrame, shipping_cost: float = 3.0) -> pd.DataFrame:
    """Construct spread series from normalized prices.

    Args:
        norm_df: DataFrame from :func:`normalize_to_eur_mwh` containing
            ``TTF_EUR_MWh``, ``NBP_EUR_MWh``, and ``JKM_EUR_MWh``.
        shipping_cost: Assumed LNG shipping cost in EUR/MWh used when computing
            the TTF vs. JKM netback spread.

    Returns:
        ``pd.DataFrame`` with ``TTF_NBP`` and ``TTF_JKM_netback`` columns.
    """

    required = {"TTF_EUR_MWh", "NBP_EUR_MWh", "JKM_EUR_MWh"}
    missing = required - set(norm_df.columns)
    if missing:
        raise ValueError(f"Normalized dataframe missing columns: {', '.join(sorted(missing))}")

    spreads = pd.DataFrame(index=norm_df.index)
    spreads["TTF_NBP"] = norm_df["TTF_EUR_MWh"] - norm_df["NBP_EUR_MWh"]
    spreads["TTF_JKM_netback"] = norm_df["TTF_EUR_MWh"] - (norm_df["JKM_EUR_MWh"] + shipping_cost)

    return spreads


__all__ = ["normalize_to_eur_mwh", "build_spreads", "THERM_TO_MWH", "MMBTU_TO_MWH"]
