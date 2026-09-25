"""A small, honestly-evaluated machine learning model.

Predicts the DIRECTION of price movement a few days ahead using basic
technical features, trained only on one asset's own price history. It is
evaluated with a strict chronological train/test split (never shuffled), so
the reported accuracy reflects genuine out-of-sample performance, not a
model that has secretly seen the future.

Read the reported numbers next to the "naive baseline" (always predicting
the market's historical majority direction over the test window). If the
model barely beats that baseline, that's a real and common result — this is
a genuinely hard prediction problem, and a handful of technical features
feeding a small model will not solve it.
"""

from sklearn.ensemble import RandomForestClassifier

FEATURE_COLS = ["RSI", "MACD_hist", "price_vs_sma_short", "price_vs_sma_long", "atr_pct"]


def _build_features(df, horizon):
    data = df.copy()
    data["price_vs_sma_short"] = data["Close"] / data["SMA_short"] - 1
    data["price_vs_sma_long"] = data["Close"] / data["SMA_long"] - 1
    data["atr_pct"] = data["ATR"] / data["Close"]
    data["future_return"] = data["Close"].shift(-horizon) / data["Close"] - 1
    return data


def train_and_evaluate(df, horizon=5, test_fraction=0.3):
    """Train on the older portion of history, test on the newer portion only."""
    data = _build_features(df, horizon)

    # Rows with a known future outcome (needed to train/test); this
    # necessarily excludes the most recent `horizon` days.
    labeled = data.dropna(subset=FEATURE_COLS + ["future_return"]).copy()
    labeled["target"] = (labeled["future_return"] > 0).astype(int)

    if len(labeled) < 120:
        return None

    split_idx = int(len(labeled) * (1 - test_fraction))
    train, test = labeled.iloc[:split_idx], labeled.iloc[split_idx:]
    if len(test) < 20 or train["target"].nunique() < 2:
        return None

    model = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=10, random_state=42
    )
    model.fit(train[FEATURE_COLS], train["target"])

    predictions = model.predict(test[FEATURE_COLS])
    accuracy = (predictions == test["target"]).mean()
    baseline_accuracy = max(test["target"].mean(), 1 - test["target"].mean())

    # Today's row: features are known, outcome isn't yet — that's the point.
    latest_row = data.dropna(subset=FEATURE_COLS).iloc[[-1]]
    latest_pred = model.predict(latest_row[FEATURE_COLS])[0]
    latest_proba = model.predict_proba(latest_row[FEATURE_COLS])[0][latest_pred]

    return {
        "test_accuracy": round(float(accuracy) * 100, 1),
        "baseline_accuracy": round(float(baseline_accuracy) * 100, 1),
        "edge_over_baseline": round(float(accuracy - baseline_accuracy) * 100, 1),
        "latest_prediction": "UP" if latest_pred == 1 else "DOWN",
        "latest_confidence": round(float(latest_proba) * 100, 1),
        "horizon_days": horizon,
        "test_samples": len(test),
    }
