from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_agent.indicators import rsi


def test_rsi_returns_same_length_as_input() -> None:
    prices = pd.Series(np.linspace(100, 130, 20))
    values = rsi(prices, period=14)
    assert len(values) == len(prices)


def test_rsi_raises_on_too_short_input() -> None:
    with pytest.raises(ValueError):
        rsi([1, 2, 3, 4], period=14)


def test_rsi_values_are_within_bounds_when_not_nan() -> None:
    prices = np.linspace(100, 130, 30) + np.sin(np.linspace(0, 6, 30)) * 5
    values = rsi(prices, period=14)
    non_nan = values.dropna()
    assert (non_nan >= 0).all()
    assert (non_nan <= 100).all()

