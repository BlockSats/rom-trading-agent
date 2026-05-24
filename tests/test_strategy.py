from __future__ import annotations

import inspect

import pandas as pd

from trading_agent.data import generate_sample_ohlcv
from trading_agent.strategy import run_strategy_on_dataframe
import trading_agent.broker_paper as broker_paper


def test_sample_data_can_run_through_strategy() -> None:
    strategy = {
        "version": "0001",
        "asset": "BTC/USDT",
        "timeframe": "1h",
        "entry": {"indicator": "rsi", "threshold": 30, "direction": "long"},
        "exit": {"rsi_take_profit": 55},
        "risk": {"stop_loss_pct": 2.0, "position_size_pct": 10.0},
        "costs": {"fee_pct": 0.1, "slippage_pct": 0.05},
        "reflection": {"one_variable_only": True, "allowed_variables": ["entry.threshold"]},
    }
    df = generate_sample_ohlcv(rows=50, seed=42)
    trades = run_strategy_on_dataframe(df, strategy)
    assert isinstance(trades, list)


def test_generated_sample_data_is_deterministic_with_same_seed() -> None:
    first = generate_sample_ohlcv(rows=25, seed=7)
    second = generate_sample_ohlcv(rows=25, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_no_live_trading_components_exist() -> None:
    source = inspect.getsource(broker_paper).lower()
    forbidden = ["ccxt", "binance", "kraken", "websocket", "api key", "private key"]
    assert not any(token in source for token in forbidden)

