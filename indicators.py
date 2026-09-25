"""Technical indicator calculations, computed from plain OHLC data.

Every indicator here is a rolling/backward-looking function of past prices
only (standard pandas .rolling()/.ewm() behaviour) — none of them peek at
future data, which matters for the backtester and ML model to be honest.
"""

import numpy as np
import pandas as pd


def add_all_indicators(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
    rsi_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20,
    bb_std: float = 2.0,
    atr_window: int = 14,
) -> pd.DataFrame:
    """Return a copy of df with technical indicator columns added."""
    out = df.copy()

    out["SMA_short"] = out["Close"].rolling(short_window).mean()
    out["SMA_long"] = out["Close"].rolling(long_window).mean()

    # --- RSI ---
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_window).mean()
    avg_loss = loss.rolling(rsi_window).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

    # avg_loss == 0 makes rs undefined/inf: RSI is 100 if there were gains,
    # or 50 (neutral) if price didn't move at all.
    rsi = np.where(avg_loss == 0, np.where(avg_gain == 0, 50.0, 100.0), rsi)
    out["RSI"] = pd.Series(rsi, index=out.index).fillna(50.0)

    # --- MACD ---
    ema_fast = out["Close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = out["Close"].ewm(span=macd_slow, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_signal"] = out["MACD"].ewm(span=macd_signal, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    # --- Bollinger Bands ---
    mid = out["Close"].rolling(bb_window).mean()
    std = out["Close"].rolling(bb_window).std()
    out["BB_mid"] = mid
    out["BB_upper"] = mid + bb_std * std
    out["BB_lower"] = mid - bb_std * std

    # --- ATR (simplified: rolling mean of True Range, not Wilder-smoothed) ---
    high_low = out["High"] - out["Low"]
    high_close = (out["High"] - out["Close"].shift()).abs()
    low_close = (out["Low"] - out["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["ATR"] = true_range.rolling(atr_window).mean()

    return out
