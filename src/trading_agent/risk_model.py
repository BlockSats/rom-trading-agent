from __future__ import annotations

import math


def compute_atr_position_size(
    equity: float,
    price: float,
    atr_value: float,
    risk_pct: float,
    atr_stop_multiplier: float,
) -> float:
    """Return position size as a percentage of equity (0–100).

    Sizes the position so that a stop at atr_value * atr_stop_multiplier below
    entry would lose exactly risk_pct percent of equity.
    """
    if equity <= 0:
        raise ValueError("equity must be positive")
    if price <= 0:
        raise ValueError("price must be positive")
    if math.isnan(atr_value) or atr_value <= 0:
        raise ValueError("atr_value must be positive")
    if risk_pct <= 0:
        raise ValueError("risk_pct must be positive")
    if atr_stop_multiplier <= 0:
        raise ValueError("atr_stop_multiplier must be positive")

    stop_distance = atr_value * atr_stop_multiplier
    risk_amount = equity * (risk_pct / 100.0)
    units = risk_amount / stop_distance
    position_value = units * price
    size_pct = (position_value / equity) * 100.0
    return min(size_pct, 100.0)


def compute_atr_stop_price(
    entry_price: float,
    atr_value: float,
    atr_stop_multiplier: float,
) -> float:
    """Return ATR-based stop loss price, floored at zero."""
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if math.isnan(atr_value) or atr_value <= 0:
        raise ValueError("atr_value must be positive")
    if atr_stop_multiplier <= 0:
        raise ValueError("atr_stop_multiplier must be positive")

    stop_price = entry_price - atr_value * atr_stop_multiplier
    return max(stop_price, 0.0)
