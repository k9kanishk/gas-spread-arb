from pathlib import Path

import yfinance as yf

raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)

tickers = ["EURUSD=X", "GBPUSD=X"]
df = yf.download(tickers, start="2015-01-01", progress=False)["Close"]
df.columns = ["EURUSD", "GBPUSD"]
df.index.name = "Date"
df.to_csv(raw_dir / "fx_rates.csv")

print("Saved data/raw/fx_rates.csv")
print(df.tail())
