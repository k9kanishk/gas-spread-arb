import pandas as pd
import yfinance as yf
from pathlib import Path

# Make sure raw data folder exists
raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)

# Download FX
tickers = ["EURUSD=X", "GBPUSD=X"]
fx = yf.download(tickers, start="2015-01-01")["Adj Close"]

# Rename columns
fx.columns = ["EURUSD", "GBPUSD"]

# Save to CSV
out_path = raw_dir / "fx_rates.csv"
fx.to_csv(out_path)

print(f"Saved FX data to {out_path}")
print(fx.tail())
