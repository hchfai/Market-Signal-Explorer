"""Batch screener: run signals across popular assets, rank by signal strength."""

import time
from datetime import datetime

import pandas as pd

from data_crypto import SYMBOL_TO_ID, get_crypto_history
from data_stocks import get_stock_history
from indicators import add_all_indicators
from signal_engine import generate_signal

# Top 50+ ASX stocks by market cap + popular ETFs
ASX_TICKERS = [
    "CBA.AX", "BHP.AX", "NAB.AX", "ANZ.AX", "WBC.AX", "TLS.AX", "WES.AX", "MQG.AX",
    "AMP.AX", "APT.AX", "ASX.AX", "APA.AX", "AWC.AX", "BXB.AX", "COL.AX", "CSL.AX",
    "DXN.AX", "FMG.AX", "GMG.AX", "GUD.AX", "IAG.AX", "JHX.AX", "MND.AX", "MPL.AX",
    "NWL.AX", "ORE.AX", "ORI.AX", "PMV.AX", "QAN.AX", "REH.AX", "RMD.AX", "SCG.AX",
    "SUN.AX", "TCL.AX", "TPM.AX", "TWE.AX", "VCX.AX", "VEA.AX", "VGS.AX", "VAS.AX",
    "VOC.AX", "WAM.AX", "WPL.AX", "XJO.AX", "YOW.AX", "Z1P.AX",
]

# Top 50+ US stocks (mega-cap, large-cap, growth)
US_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "HD", "NFLX", "DIS", "ADBE", "CRM", "PYPL", "IBM",
    "INTC", "AMD", "QCOM", "CSCO", "ORCL", "AVGO", "ASML", "NOW", "SHOP", "INTU",
    "SQ", "ROKU", "SNAP", "TWTR", "PINS", "ZM", "DKNG", "DASH", "UBER", "LYFT",
    "NIO", "PLTR", "COIN", "MSTR", "RIOT", "MARA", "CLSK", "HUT", "F", "GM",
    "BA", "MMM", "CAT", "DE", "GE", "HON",
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
            df = get_stock_history(ticker, days=365, retries=1)
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
    categories: list of ("stock_asx", "stock_us", "crypto") or None for all
    limit: max results per category or None for all
    newsapi_key: optional key to fetch sentiment (improves signal quality)
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
        row = _screen_ticker(ticker, asset_type, newsapi_key=newsapi_key)
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
