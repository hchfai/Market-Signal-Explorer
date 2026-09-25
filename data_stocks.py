"""Fetch historical stock price data from Yahoo Finance via yfinance.

Yahoo Finance (which yfinance wraps) rate-limits aggressively at times and
occasionally changes its internal API, which briefly breaks yfinance until
it's updated. This wraps fetches in a short retry/backoff and fails
gracefully (returns None) rather than crashing the app.
"""

import time
from datetime import datetime, timedelta

import yfinance as yf


def get_stock_history(ticker, days=365, retries=2):
    """Return a DataFrame with Open/High/Low/Close/Volume, or None on failure."""
    end = datetime.now()
    start = end - timedelta(days=int(days) + 5)
    symbol = ticker.strip().upper()

    for attempt in range(retries + 1):
        try:
            data = yf.Ticker(symbol).history(
                start=start, end=end, interval="1d", auto_adjust=True
            )
            if data is not None and not data.empty:
                return data[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception:
            pass
        if attempt < retries:
            time.sleep(2 * (attempt + 1))

    return None
