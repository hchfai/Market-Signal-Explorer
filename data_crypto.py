"""Fetch historical crypto price data from CoinGecko's public API.

Works with no API key at all (CoinGecko's free tier allows this at a lower
rate limit, ~30 calls/minute). An optional free "Demo" key raises that limit
a bit if you add one to .env.
"""

import requests
import pandas as pd

SYMBOL_TO_ID = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "ADA": "cardano",
    "XRP": "ripple", "DOGE": "dogecoin", "DOT": "polkadot", "MATIC": "matic-network",
    "LTC": "litecoin", "AVAX": "avalanche-2", "LINK": "chainlink", "BNB": "binancecoin",
    "USDT": "tether", "USDC": "usd-coin", "SHIB": "shiba-inu", "TRX": "tron",
}


def get_crypto_history(coin_id, vs_currency="aud", days=365, api_key=None):
    """Return a daily OHLC(-ish) DataFrame for a CoinGecko coin id, or None on failure.

    CoinGecko's free tier caps historical range at ~365 days, so `days` is
    clamped to that regardless of what's requested.
    """
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": vs_currency, "days": min(int(days), 365)}
        headers = {"x-cg-demo-api-key": api_key} if api_key else {}

        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        prices = payload.get("prices", [])
        if not prices:
            return None

        series = pd.DataFrame(prices, columns=["timestamp", "price"])
        series["Date"] = pd.to_datetime(series["timestamp"], unit="ms")
        series = series.set_index("Date")["price"]

        daily = series.resample("D").agg(["first", "max", "min", "last"]).dropna()
        daily.columns = ["Open", "High", "Low", "Close"]
        daily["Volume"] = 0
        return daily
    except Exception:
        return None
