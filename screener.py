"""Batch screener: run signals across popular assets, rank by signal strength."""

import time
from datetime import datetime

import pandas as pd

from data_crypto import SYMBOL_TO_ID, get_crypto_history
from data_stocks import get_stock_history
from indicators import add_all_indicators
from signal_engine import generate_signal

# Popular ASX stocks
ASX_TICKERS = [
    "CBA.AX",  # Commonwealth Bank
    "BHP.AX",  # BHP Billiton
    "NAB.AX",  # NAB
    "ANZ.AX",  # ANZ
    "WBC.AX",  # Westpac
    "TLS.AX",  # Telstra
    "WES.AX",  # Wesfarmers
    "MQG.AX",  # Macquarie
    "VAS.AX",  # Vanguard All-Ordinaries
    "VGS.AX",  # Vanguard Global Shares
]

# Popular US stocks
US_TICKERS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL",  # Google
    "AMZN",  # Amazon
    "TSLA",  # Tesla
    "NVDA",  # Nvidia
    "META",  # Meta
    "JPM",  # JPMorgan
    "V",  # Visa
    "JNJ",  # Johnson & Johnson
]

# Popular crypto
CRYPTO_SYMBOLS = ["BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "AVAX", "LINK"]


def _screen_ticker(ticker, asset_type="stock"):
    """Try to fetch and signal one ticker; return row for results df, or None if fails."""
    try:
        if asset_type == "stock":
            df = get_stock_history(ticker, days=365, retries=1)
        else:
            coin_id = SYMBOL_TO_ID.get(ticker, ticker.lower())
            df = get_crypto_history(coin_id, days=365)

        if df is None or len(df) < 60:
            return None

        df = add_all_indicators(df)
        latest, prev = df.iloc[-1], df.iloc[-2]
        signal = generate_signal(latest, prev, sentiment_score=None)

        return {
            "ticker": ticker,
            "asset_type": asset_type,
            "price": round(latest["Close"], 4),
            "signal": signal["action"],
            "confidence": signal["confidence"],
            "score": signal["score"],
        }
    except Exception:
        return None


def run_screener(categories=None, limit=None):
    """
    Scan popular assets, return a sorted DataFrame of results.
    categories: list of ("stock_asx", "stock_us", "crypto") or None for all
    limit: max results per category or None for all
    """
    if categories is None:
        categories = ["stock_asx", "stock_us", "crypto"]

    results = []
    tickers_to_check = []

    if "stock_asx" in categories:
        tickers_to_check.extend([(t, "stock") for t in ASX_TICKERS])
    if "stock_us" in categories:
        tickers_to_check.extend([(t, "stock") for t in US_TICKERS])
    if "crypto" in categories:
        tickers_to_check.extend([(t, "crypto") for t in CRYPTO_SYMBOLS])

    for ticker, asset_type in tickers_to_check:
        row = _screen_ticker(ticker, asset_type)
        if row:
            results.append(row)
        time.sleep(0.2)  # Rate-limit politeness

    if not results:
        return None

    df = pd.DataFrame(results)

    signal_order = {"BUY": 0, "SELL": 1, "HOLD": 2}
    confidence_order = {"High": 0, "Medium": 1, "Low": 2}
    df["_signal_order"] = df["signal"].map(signal_order)
    df["_confidence_order"] = df["confidence"].map(confidence_order)
    df = df.sort_values(
        ["_signal_order", "_confidence_order", "score"], ascending=[True, True, False]
    ).drop(columns=["_signal_order", "_confidence_order"])

    if limit:
        df = df.head(limit)

    return df.reset_index(drop=True)
