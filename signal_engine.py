"""Combine technical indicators and news sentiment into a single trade signal.

This is a heuristic, not a prediction. It is one input, not an answer.
"""

DEFAULT_WEIGHTS = {"trend": 0.35, "macd": 0.25, "rsi": 0.15, "sentiment": 0.25}


def generate_signal(latest, prev, sentiment_score=None, weights=None):
    """
    latest, prev: pandas Series (consecutive rows) with indicator columns
                  already present (SMA_short, SMA_long, RSI, MACD,
                  MACD_signal, Close).
    sentiment_score: float in [-1, 1], or None if unavailable.

    Returns a dict with action, confidence, score, and plain-English reasons.
    """
    weights = weights or DEFAULT_WEIGHTS
    reasons = []

    trend_score = 0
    if latest["Close"] > latest["SMA_short"] > latest["SMA_long"]:
        trend_score = 1
        reasons.append("Price is in an uptrend (above both moving averages)")
    elif latest["Close"] < latest["SMA_short"] < latest["SMA_long"]:
        trend_score = -1
        reasons.append("Price is in a downtrend (below both moving averages)")

    macd_score = 0
    if prev["MACD"] <= prev["MACD_signal"] and latest["MACD"] > latest["MACD_signal"]:
        macd_score = 1
        reasons.append("MACD just turned bullish")
    elif prev["MACD"] >= prev["MACD_signal"] and latest["MACD"] < latest["MACD_signal"]:
        macd_score = -1
        reasons.append("MACD just turned bearish")

    rsi_score = 0
    if latest["RSI"] < 30:
        rsi_score = 1
        reasons.append(f"RSI shows oversold conditions ({latest['RSI']:.0f})")
    elif latest["RSI"] > 70:
        rsi_score = -1
        reasons.append(f"RSI shows overbought conditions ({latest['RSI']:.0f})")

    sent_score = 0.0
    if sentiment_score is not None:
        sent_score = max(-1.0, min(1.0, sentiment_score))
        if sent_score > 0.2:
            reasons.append("Recent news sentiment is positive")
        elif sent_score < -0.2:
            reasons.append("Recent news sentiment is negative")

    composite = (
        weights["trend"] * trend_score
        + weights["macd"] * macd_score
        + weights["rsi"] * rsi_score
        + weights["sentiment"] * sent_score
    )

    if composite >= 0.4:
        action = "BUY"
    elif composite <= -0.4:
        action = "SELL"
    else:
        action = "HOLD"

    confidence = "Low"
    if abs(composite) >= 0.7:
        confidence = "High"
    elif abs(composite) >= 0.4:
        confidence = "Medium"

    if not reasons:
        reasons.append("No strong technical or sentiment signal right now")

    return {
        "action": action,
        "confidence": confidence,
        "score": round(float(composite), 2),
        "reasons": reasons,
    }
