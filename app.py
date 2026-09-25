"""Market Signal Explorer — Streamlit dashboard with screener + deep dive."""

import re

import plotly.graph_objects as go
import streamlit as st

from backtester import run_backtest
from config import load_config
from data_crypto import SYMBOL_TO_ID, get_crypto_history
from data_stocks import get_stock_history
from indicators import add_all_indicators
from ml_model import train_and_evaluate
from risk_manager import (
    diversification_check,
    position_size,
    suggested_stop_loss,
    suggested_take_profit,
)
from sentiment import get_news_sentiment
from signal_engine import generate_signal

MIN_ROWS = 60
TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]{1,20}$")

st.set_page_config(page_title="Market Signal Explorer", page_icon="📊", layout="wide")
cfg = load_config()


def format_price(x):
    if x is None:
        return "n/a"
    if x >= 1:
        return f"{x:,.2f}"
    if x >= 0.01:
        return f"{x:.4f}"
    return f"{x:.8f}"


st.title("📊 Market Signal Explorer")
st.warning(
    "**Educational tool, not financial advice.** No app can reliably predict "
    "markets. Signals are heuristics from historical price patterns and "
    "recent news sentiment, not certainties. Past performance does not "
    "indicate future results, and you can lose some or all of the money "
    "you use with this."
)

tab_screener, tab_analyze = st.tabs(["📈 Today's Signals", "🔍 Deep Dive"])

# ============================================================================
# SCREENER TAB
# ============================================================================
with tab_screener:
    st.subheader("Today's Best Signals")
    st.caption(
        "Browse popular stocks and crypto. The app runs signals on each one "
        "and ranks them by strength. Click any ticker to load a full analysis."
    )

    col1, col2 = st.columns(2)
    with col1:
        check_asx = st.checkbox("ASX (Australian)", value=True)
        check_us = st.checkbox("US stocks", value=True)
    with col2:
        check_crypto = st.checkbox("Crypto", value=True)

    if st.button("🔄 Run Screener", type="primary"):
        categories = []
        if check_asx:
            categories.append("stock_asx")
        if check_us:
            categories.append("stock_us")
        if check_crypto:
            categories.append("crypto")

        if not categories:
            st.error("Pick at least one category.")
        else:
            with st.spinner("Scanning popular assets... this takes ~30 seconds"):
                from screener import run_screener

                results = run_screener(categories=categories, limit=20)

            if results is None:
                st.warning(
                    "Screener had trouble fetching data. "
                    "Try again in a minute — rate limits might have kicked in."
                )
            else:
                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "ticker": st.column_config.TextColumn("Ticker", width="small"),
                        "asset_type": st.column_config.TextColumn("Type", width="small"),
                        "price": st.column_config.NumberColumn("Price", format="%.4f"),
                        "signal": st.column_config.TextColumn("Signal", width="small"),
                        "confidence": st.column_config.TextColumn("Confidence", width="small"),
                        "score": st.column_config.NumberColumn("Score", format="%.2f"),
                    },
                )

                st.markdown("**Copy a ticker from above**, paste it in the Deep Dive tab → Analyze.")

# ============================================================================
# DEEP DIVE TAB
# ============================================================================
with tab_analyze:
    st.subheader("Analyze an Asset")

    with st.sidebar:
        st.header("Settings")
        asset_type = st.radio("Asset type", ["Stock", "Crypto"])
        if asset_type == "Stock":
            symbol = st.text_input("Ticker (e.g. AAPL, CBA.AX, VAS.AX)", value="AAPL")
        else:
            symbol = st.text_input("Coin symbol (e.g. BTC, ETH, SOL)", value="BTC")
        period_days = st.slider("History (days)", 90, 730, 365)
        starting_capital = st.number_input(
            "Starting capital (AUD)", min_value=10.0, value=500.0, step=10.0
        )
        risk_pct = st.slider("Risk per trade (%)", 0.5, 5.0, 2.0, 0.5)
        run_clicked = st.button("Analyze", type="primary")

    if not run_clicked:
        st.info("Set your options in the sidebar and click **Analyze**.")
        st.stop()

    symbol = symbol.strip()
    if not TICKER_RE.match(symbol):
        st.error(
            "That doesn't look like a valid ticker/symbol. Use letters, numbers, '.' or '-' only."
        )
        st.stop()

    with st.spinner("Fetching price data..."):
        if asset_type == "Stock":
            df = get_stock_history(symbol, period_days)
            label = symbol.upper()
            currency_note = (
                "Prices are in the ticker's native currency "
                "(e.g. USD for US tickers, AUD for .AX tickers)."
            )
        else:
            coin_id = SYMBOL_TO_ID.get(symbol.upper(), symbol.lower())
            df = get_crypto_history(
                coin_id, days=period_days, api_key=cfg.get("COINGECKO_API_KEY")
            )
            label = symbol.upper()
            currency_note = (
                "Crypto prices are shown in AUD. CoinGecko's free tier provides "
                "up to about a year of history."
            )

    if df is None or len(df) < MIN_ROWS:
        st.error(
            f"Couldn't get enough data for '{symbol}'. Check the symbol, try a "
            "longer history window, or try again in a minute — data providers "
            "rate-limit requests."
        )
        st.stop()

    st.caption(currency_note)
    df = add_all_indicators(df)
    latest, prev = df.iloc[-1], df.iloc[-2]

    sentiment_score, headlines = None, []
    if cfg.get("NEWSAPI_KEY"):
        with st.spinner("Checking recent news sentiment..."):
            sentiment_score, headlines = get_news_sentiment(label, cfg["NEWSAPI_KEY"])

    signal = generate_signal(latest, prev, sentiment_score)

    price_col, signal_col = st.columns([2, 1])

    with price_col:
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=label,
            )
        )
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_short"], name="SMA 20", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_long"], name="SMA 50", line=dict(width=1)))
        fig.update_layout(height=480, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with signal_col:
        st.markdown(f"**{label}** — current price: {format_price(latest['Close'])}")
        color = {"BUY": "green", "SELL": "red", "HOLD": "gray"}[signal["action"]]
        st.markdown(f"### Signal: :{color}[{signal['action']}]")
        st.markdown(f"**Confidence:** {signal['confidence']}  \n**Score:** {signal['score']}")
        st.markdown("**Why:**")
        for reason in signal["reasons"]:
            st.markdown(f"- {reason}")

        if signal["action"] == "BUY":
            atr = latest["ATR"]
            entry = latest["Close"]
            stop = suggested_stop_loss(entry, atr)
            target = suggested_take_profit(entry, atr)
            units, cost = position_size(starting_capital, entry, stop, risk_pct)
            pct, warn = diversification_check(cost, starting_capital)

            st.markdown("**If you were to enter here:**")
            st.markdown(f"- Suggested stop-loss: {format_price(stop)}")
            st.markdown(f"- Suggested take-profit: {format_price(target)}")
            st.markdown(f"- Size at {risk_pct}% risk: {units:g} units (~${cost:.2f})")
            if warn:
                st.info(warn)
            st.caption(
                "These are suggestions from a simple rule, not instructions. "
                "Round to whatever your broker/exchange allows."
            )

    st.divider()
    st.subheader("Recent news feeding the sentiment score")
    if headlines:
        for h in headlines[:5]:
            st.markdown(f"- [{h['title']}]({h['url']}) — sentiment {h['score']:.2f}")
    elif cfg.get("NEWSAPI_KEY"):
        st.caption("No recent headlines found for this asset.")
    else:
        st.caption(
            "Add a free NEWSAPI_KEY to your .env to include news sentiment (see README.md)."
        )

    st.divider()
    st.subheader("Machine learning read (experimental)")
    st.caption(
        "A small model trained only on this asset's own recent history, tested "
        "on data it never saw during training. Compare its accuracy to the "
        "naive baseline below — when they're close, the model isn't adding "
        "much, which is a common, honest result in short-term price prediction."
    )
    with st.spinner("Training and evaluating model..."):
        ml_result = train_and_evaluate(df)

    if ml_result is None:
        st.caption("Not enough data or variation to train a reliable model for this asset/period.")
    else:
        st.markdown(
            f"Out-of-sample accuracy: **{ml_result['test_accuracy']}%** vs a naive "
            f"baseline of **{ml_result['baseline_accuracy']}%** "
            f"(edge: {ml_result['edge_over_baseline']:+.1f} points, on "
            f"{ml_result['test_samples']} unseen days)."
        )
        st.markdown(
            f"Latest read: **{ml_result['latest_prediction']}** over the next "
            f"{ml_result['horizon_days']} trading days, "
            f"{ml_result['latest_confidence']}% model confidence."
        )
        st.caption(
            "Model confidence isn't the same as being right — treat this as one "
            "more data point, not an answer."
        )

    st.divider()
    st.subheader("Backtest (technical rules only)")
    st.caption(
        "News sentiment isn't included here — free news APIs don't provide "
        "enough history to backtest it fairly. This shows how the "
        "price/momentum rules alone would have performed, starting from your "
        "stated capital, ignoring fees and slippage."
    )
    with st.spinner("Running backtest..."):
        result = run_backtest(df, starting_capital=starting_capital, risk_per_trade_pct=risk_pct)

    if result is None:
        st.warning("Not enough history for a meaningful backtest — try a longer window.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Strategy return", f"{result['total_return_pct']:.1f}%")
        m2.metric("Buy & hold return", f"{result['buy_hold_return_pct']:.1f}%")
        m3.metric("Max drawdown", f"{result['max_drawdown_pct']:.1f}%")
        m4.metric("Win rate", f"{result['win_rate_pct']:.0f}% ({result['num_trades']} trades)")
        st.line_chart(result["equity_curve"]["equity"])
        st.caption(
            "Notice this is nowhere near turning $500 into $100k in a month — "
            "that's the honest, realistic picture, not a limitation of this "
            "specific tool."
        )
