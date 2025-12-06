import pandas as pd
import yfinance as yf
from pathlib import Path

# Make sure folders exist
Path("data/raw").mkdir(parents=True, exist_ok=True)

START_DATE = "2018-01-01"

tickers = {
    "TTF": "TTF=F",        # Dutch TTF Natural Gas
    "NBP": "E2X16.NYM",    # National Balancing Point (UK gas)
}

for name, ticker in tickers.items():
    print(f"Downloading {name} from {ticker}...")
    df = yf.download(ticker, start=START_DATE)
    # Use adjusted close as price
    out = df[["Adj Close"]].rename(columns={"Adj Close": "Price"})
    out.to_csv(f"data/raw/{name.lower()}_prices.csv")
    print(f"Saved data/raw/{name.lower()}_prices.csv with {len(out)} rows.")

# FX
print("Downloading FX (EURUSD, GBPUSD)...")
fx_tickers = ["EURUSD=X", "GBPUSD=X"]
fx = yf.download(fx_tickers, start=START_DATE)["Adj Close"]
fx.columns = ["EURUSD", "GBPUSD"]
fx.to_csv("data/raw/fx_rates.csv")
print(f"Saved data/raw/fx_rates.csv with {len(fx)} rows.")
