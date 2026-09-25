"""Simple historical backtest of the technical signal rules.

News sentiment is intentionally excluded: free news APIs don't provide
enough historical depth to backtest that part honestly. This shows how the
price/momentum rules alone would have performed. Treat the results as a
reality check, not a promise — and note real trading also has fees and
slippage, which this simplified version does not model.
"""

import pandas as pd

from risk_manager import position_size, suggested_stop_loss, suggested_take_profit
from signal_engine import generate_signal

WARMUP = 55  # bars needed before indicators (SMA_long=50 etc.) are reliable


def run_backtest(df, starting_capital=500.0, risk_per_trade_pct=2.0,
                  stop_atr_mult=2.0, target_atr_mult=4.0):
    if df is None or len(df) <= WARMUP + 5:
        return None

    cash = starting_capital
    position = None
    equity_curve = []
    trades = []

    for i in range(WARMUP, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        date = df.index[i]
        signal = generate_signal(row, prev_row, sentiment_score=None)

        if position is not None:
            exit_price = None
            if row["Low"] <= position["stop_loss"]:
                exit_price = position["stop_loss"]
            elif row["High"] >= position["take_profit"]:
                exit_price = position["take_profit"]
            elif signal["action"] == "SELL":
                exit_price = row["Close"]

            if exit_price is not None:
                proceeds = position["shares"] * exit_price
                pnl = proceeds - position["cost"]
                cash += proceeds
                trades.append({
                    "entry_date": position["entry_date"],
                    "exit_date": date,
                    "pnl": pnl,
                    "pnl_pct": (pnl / position["cost"] * 100) if position["cost"] else 0.0,
                })
                position = None

        if position is None and signal["action"] == "BUY":
            atr = row.get("ATR")
            if atr and atr > 0:
                entry_price = row["Close"]
                stop = suggested_stop_loss(entry_price, atr, stop_atr_mult)
                target = suggested_take_profit(entry_price, atr, target_atr_mult)
                units, cost = position_size(cash, entry_price, stop, risk_per_trade_pct)
                if units > 0 and cost > 0:
                    cash -= cost
                    position = {
                        "entry_date": date,
                        "shares": units,
                        "cost": cost,
                        "stop_loss": stop,
                        "take_profit": target,
                    }

        equity = cash + (position["shares"] * row["Close"] if position else 0)
        equity_curve.append({"date": date, "equity": equity})

    if position is not None:
        last_row = df.iloc[-1]
        proceeds = position["shares"] * last_row["Close"]
        pnl = proceeds - position["cost"]
        cash += proceeds
        trades.append({
            "entry_date": position["entry_date"],
            "exit_date": df.index[-1],
            "pnl": pnl,
            "pnl_pct": (pnl / position["cost"] * 100) if position["cost"] else 0.0,
        })

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    final_equity = equity_df["equity"].iloc[-1] if not equity_df.empty else starting_capital
    total_return_pct = (final_equity / starting_capital - 1) * 100

    running_max = equity_df["equity"].cummax()
    drawdown_pct = (equity_df["equity"] - running_max) / running_max * 100
    max_drawdown_pct = drawdown_pct.min() if not drawdown_pct.empty else 0.0

    wins = [t for t in trades if t["pnl"] > 0]
    win_rate_pct = (len(wins) / len(trades) * 100) if trades else 0.0

    start_price = df.iloc[WARMUP]["Close"]
    end_price = df.iloc[-1]["Close"]
    buy_hold_return_pct = (end_price / start_price - 1) * 100

    return {
        "equity_curve": equity_df,
        "trades": trades,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate_pct": win_rate_pct,
        "num_trades": len(trades),
        "buy_hold_return_pct": buy_hold_return_pct,
        "final_equity": final_equity,
    }
