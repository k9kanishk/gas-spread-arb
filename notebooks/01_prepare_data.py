"""
Example notebook code for preparing gas price data.

Copy/paste these cells into a Jupyter notebook (or run directly via
`jupytext`) to load raw CSVs, normalize them to EUR/MWh, build spreads,
and visualize the results.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt

from src.data_loader import load_raw_prices, save_clean_prices
from src.spreads import build_spreads, normalize_to_eur_mwh

# %% [markdown]
# ## Load raw price and FX data
#
# Raw CSVs are expected under `data/raw` with the following filenames:
# - `ttf_prices.csv`
# - `nbp_prices.csv`
# - `jkm_prices.csv`
# - `fx_rates.csv`
#
# Each file needs a `Date` column. Price columns can be named `Price` or any
# other non-date column.

# %%
raw_df = load_raw_prices()
raw_df.head()

# %% [markdown]
# ## Save the cleaned dataset
# Forward-filled and aligned prices are saved to
# `data/processed/clean_prices.csv` for reproducibility.

# %%
clean_path = save_clean_prices(raw_df)
print(f"Saved cleaned prices to: {clean_path}")

# %% [markdown]
# ## Normalize hub prices to EUR/MWh
# Convert TTF, NBP, and JKM prices into consistent EUR/MWh units.

# %%
normalized_df = normalize_to_eur_mwh(raw_df)
normalized_path = Path("data/processed/normalized_prices.csv")
normalized_df.to_csv(normalized_path)
print(f"Saved normalized prices to: {normalized_path}")

# %% [markdown]
# ## Build TTF vs. NBP and TTF vs. JKM netback spreads
# The default LNG shipping cost of EUR 3.0/MWh is used when constructing the
# TTF vs. JKM netback spread.

# %%
spreads_df = build_spreads(normalized_df)
spreads_path = Path("data/processed/spreads.csv")
spreads_df.to_csv(spreads_path)
print(f"Saved spreads to: {spreads_path}")

# %% [markdown]
# ## Quick visual inspection
# Plot normalized hub prices and the resulting spreads.

# %%
plt.style.use("ggplot")

fig, ax = plt.subplots(figsize=(10, 5))
normalized_df.plot(ax=ax)
ax.set_title("Normalized hub prices (EUR/MWh)")
ax.set_ylabel("EUR/MWh")
ax.legend(loc="upper left")
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
spreads_df.plot(ax=ax)
ax.set_title("Gas hub spreads (EUR/MWh)")
ax.set_ylabel("EUR/MWh")
ax.legend(loc="upper left")
plt.show()
