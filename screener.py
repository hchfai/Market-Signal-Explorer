"""Batch screener: run signals across popular assets, rank by signal strength."""

import time
from datetime import datetime

import pandas as pd

from data_crypto import SYMBOL_TO_ID, get_crypto_history
from data_stocks import get_stock_history
from indicators import add_all_indicators
from signal_engine import generate_signal

# Optimized ticker lists - fewer, more reliable tickers to avoid API rate limits

# Top 20 Wall Street / US stocks (most liquid, reliable)
US_WALLSTREET_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "NFLX", "DIS", "INTC", "AMD", "PLTR", "COIN", "MARA",
]

# Top 15 Canadian stocks (TSX, most reliable)
CANADIAN_TICKERS = [
    "TD.TO", "RY.TO", "BN.TO", "ENB.TO", "TRP.TO", "BCE.TO", "CM.TO", "BMO.TO",
    "SU.TO", "CNQ.TO", "SHOP.TO", "WES.TO", "MG.TO", "AQN.TO", "RCI.TO",
]

# Top 20 Australian stocks (ASX, most reliable, liquid)
ASX_TICKERS = [
    "CBA.AX", "BHP.AX", "NAB.AX", "ANZ.AX", "WBC.AX", "TLS.AX", "WES.AX", "MQG.AX",
    "APT.AX", "CSL.AX", "FMG.AX", "IAG.AX", "QAN.AX", "RMD.AX", "SUN.AX", "TWE.AX",
    "VGS.AX", "VAS.AX", "WPL.AX", "Z1P.AX",
]

# Global/International stocks (15 most liquid)
GLOBAL_TICKERS = [
    "SAP", "ASML", "NOVO", "BABA", "TSM", "NVO", "RHHBY", "HSBC",
    "AZN", "BAIDU", "JD", "CTRP", "TCEHY", "BKNG", "VALE",
]

# Top 40+ crypto by market cap
CRYPTO_SYMBOLS = [
    "BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "AVAX", "LINK", "DOT", "MATIC",
    "LTC", "TRX", "BCH", "SHIB", "UNI", "ATOM", "XLM", "ALGO", "NEAR", "ARB",
    "OP", "ARBITRUM", "BASE", "GALA", "MANA", "SAND", "ENS", "STX", "ICP", "FLOW",
    "FIL", "THETA", "VET", "EOS", "TEZOS", "APTOS", "SEI", "SUI", "BLUR", "GMX",
]


def _screen_ticker(ticker, asset_type="stock", newsapi_key=None):
    """Try to fetch and signal one ticker; return row for results df, or None if fails."""
    try:
        if asset_type == "stock":
            df = get_stock_history(ticker, days=365, retries=2)
        else:
            coin_id = SYMBOL_TO_ID.get(ticker, ticker.lower())
            df = get_crypto_history(coin_id, days=365)

        if df is None or len(df) < 60:
            return None

        df = add_all_indicators(df)
        latest, prev = df.iloc[-1], df.iloc[-2]

        # Fetch sentiment if NewsAPI key available
        sentiment_score = None
        if newsapi_key:
            from sentiment import get_news_sentiment

            sent, _ = get_news_sentiment(ticker, newsapi_key, page_size=10)
            sentiment_score = sent

        signal = generate_signal(latest, prev, sentiment_score=sentiment_score)

        # Heat map: how close to BUY (0.4) or SELL (-0.4) threshold?
        # Normalize score to 0-100 scale: -1 to 1 → 0 to 100
        heat = max(0, min(100, (signal["score"] + 1) / 2 * 100))

        return {
            "ticker": ticker,
            "asset_type": asset_type,
            "price": round(latest["Close"], 4),
            "signal": signal["action"],
            "confidence": signal["confidence"],
            "score": signal["score"],
            "heat": heat,
        }
    except Exception:
        return None


def run_screener(categories=None, limit=None, newsapi_key=None):
    """
    Scan popular assets, return a sorted DataFrame of results.
    categories: list of ("us_wallstreet", "canadian", "australian", "global", "crypto")
               or None for all
    limit: max results per category or None for all
    newsapi_key: optional key to fetch sentiment (improves signal quality)
    """
    if categories is None:
        categories = ["us_wallstreet", "canadian", "australian", "global", "crypto"]

    results = []
    tickers_to_check = []

    if "us_wallstreet" in categories:
        tickers_to_check.extend([(t, "stock") for t in US_WALLSTREET_TICKERS])
    if "canadian" in categories:
        tickers_to_check.extend([(t, "stock") for t in CANADIAN_TICKERS])
    if "australian" in categories:
        tickers_to_check.extend([(t, "stock") for t in ASX_TICKERS])
    if "global" in categories:
        tickers_to_check.extend([(t, "stock") for t in GLOBAL_TICKERS])
    if "crypto" in categories:
        tickers_to_check.extend([(t, "crypto") for t in CRYPTO_SYMBOLS])

    for ticker, asset_type in tickers_to_check:
        row = _screen_ticker(ticker, asset_type, newsapi_key=newsapi_key)
        if row:
            results.append(row)
        # Rate-limiting: longer delays for stocks, shorter for crypto
        delay = 0.8 if asset_type == "stock" else 0.2
        time.sleep(delay)

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
