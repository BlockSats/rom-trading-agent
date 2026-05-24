from __future__ import annotations

import numpy as np
import pandas as pd


def generate_sample_ohlcv(rows: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h")

    trend = np.linspace(100.0, 120.0, rows)
    cycle = np.sin(np.linspace(0, 6 * np.pi, rows)) * 18.0
    noise = rng.normal(0.0, 1.0, rows)
    close = trend + cycle + noise

    open_prices = np.empty(rows, dtype="float64")
    open_prices[0] = close[0]
    if rows > 1:
        open_prices[1:] = close[:-1] + rng.normal(0.0, 0.4, rows - 1)

    spread_high = np.abs(rng.normal(0.6, 0.2, rows))
    spread_low = np.abs(rng.normal(0.6, 0.2, rows))
    high = np.maximum(open_prices, close) + spread_high
    low = np.minimum(open_prices, close) - spread_low
    volume = rng.integers(100, 1000, rows)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
