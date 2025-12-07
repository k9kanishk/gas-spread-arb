# scripts/download_fx.py
from pathlib import Path
import pandas as pd
import yfinance as yf

raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)

tickers = ["EURUSD=X", "GBPUSD=X"]
fx = yf.download(tickers, start="2015-01-01")["Adj Close"]

fx.columns = ["EURUSD", "GBPUSD"]
fx.index.name = "Date"   # IMPORTANT so data_loader can do index_col="Date"

out_path = raw_dir / "fx_rates.csv"
fx.to_csv(out_path)

print(f"Saved FX data to {out_path}")
print(fx.tail())
