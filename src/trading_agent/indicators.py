from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def rsi(prices: Sequence[float], period: int = 14) -> pd.Series:
    series = pd.Series(prices, dtype="float64")
    if len(series) < period + 1:
        raise ValueError("not enough prices for RSI")

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))

    rsi_values = rsi_values.where(~avg_loss.eq(0), 100.0)
    rsi_values = rsi_values.where(~(avg_gain.eq(0) & avg_loss.eq(0)), 50.0)
    rsi_values = rsi_values.clip(lower=0.0, upper=100.0)
    return rsi_values.astype("float64")

