"""Position sizing and basic risk-management helpers.

Standard retail risk-management practice: risk a small, fixed percentage of
capital per trade, size the position from the distance to your stop-loss,
and avoid concentrating too much capital in one asset. None of this
guarantees a profit — it limits how much a wrong call can cost you.
"""

DEFAULT_RISK_PER_TRADE_PCT = 2.0
DEFAULT_MAX_ASSET_PCT = 25.0
DEFAULT_STOP_ATR_MULT = 2.0
DEFAULT_TARGET_ATR_MULT = 4.0


def suggested_stop_loss(entry_price, atr, multiplier=DEFAULT_STOP_ATR_MULT):
    return round(entry_price - multiplier * atr, 8)


def suggested_take_profit(entry_price, atr, multiplier=DEFAULT_TARGET_ATR_MULT):
    return round(entry_price + multiplier * atr, 8)


def position_size(account_balance, entry_price, stop_loss, risk_per_trade_pct=DEFAULT_RISK_PER_TRADE_PCT):
    """Return (units, cost) sized so a stop-out loses about risk_per_trade_pct of the account."""
    if account_balance <= 0 or entry_price <= 0:
        return 0.0, 0.0

    risk_amount = account_balance * (risk_per_trade_pct / 100)
    stop_distance = entry_price - stop_loss
    if stop_distance <= 0:
        return 0.0, 0.0

    units = risk_amount / stop_distance
    cost = units * entry_price
    if cost > account_balance:
        cost = account_balance
        units = cost / entry_price

    return round(units, 6), round(cost, 2)


def diversification_check(new_position_cost, account_balance, max_asset_pct=DEFAULT_MAX_ASSET_PCT):
    """Warn if a position would make up too much of the stated account balance."""
    if account_balance <= 0:
        return 0.0, None

    pct = (new_position_cost / account_balance) * 100
    warning = None
    if pct > max_asset_pct:
        warning = (
            f"This position would be about {pct:.0f}% of your stated capital. "
            f"Consider keeping any single asset under {max_asset_pct:.0f}% so one "
            f"bad trade can't hurt too much."
        )
    return pct, warning
