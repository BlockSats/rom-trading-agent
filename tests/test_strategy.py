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


def test_ema_atr_trend_can_run_through_strategy() -> None:
    strategy = {
        "version": "0001",
        "strategy_id": "ema_atr_trend",
        "asset": "BTC/USDT",
        "timeframe": "1h",
        "entry": {
            "indicator": "ema_atr",
            "fast_ema_period": 2,
            "slow_ema_period": 4,
            "direction": "long",
        },
        "exit": {"atr_period": 2, "atr_stop_multiplier": 1.5},
        "risk": {"stop_loss_pct": 2.0, "position_size_pct": 10.0},
        "costs": {"fee_pct": 0.1, "slippage_pct": 0.05},
        "reflection": {"one_variable_only": True, "allowed_variables": ["entry.fast_ema_period"]},
    }
    df = generate_sample_ohlcv(rows=50, seed=42)

    trades = run_strategy_on_dataframe(df, strategy)

    assert isinstance(trades, list)


def test_ema_atr_trend_short_history_returns_no_trades() -> None:
    strategy = {
        "version": "0001",
        "strategy_id": "ema_atr_trend",
        "asset": "BTC/USDT",
        "timeframe": "1h",
        "entry": {
            "indicator": "ema_atr",
            "fast_ema_period": 2,
            "slow_ema_period": 4,
            "direction": "long",
        },
        "exit": {"atr_period": 2, "atr_stop_multiplier": 1.5},
        "risk": {"stop_loss_pct": 2.0, "position_size_pct": 10.0},
        "costs": {"fee_pct": 0.1, "slippage_pct": 0.05},
        "reflection": {"one_variable_only": True, "allowed_variables": ["entry.fast_ema_period"]},
    }
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.0, 11.0, 12.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )

    trades = run_strategy_on_dataframe(df, strategy)

    assert trades == []


def test_generated_sample_data_is_deterministic_with_same_seed() -> None:
    first = generate_sample_ohlcv(rows=25, seed=7)
    second = generate_sample_ohlcv(rows=25, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_no_live_trading_components_exist() -> None:
    source = inspect.getsource(broker_paper).lower()
    forbidden = ["ccxt", "binance", "kraken", "websocket", "api key", "private key"]
    assert not any(token in source for token in forbidden)


def test_fixed_pct_mode_unchanged_when_mode_absent() -> None:
    """fixed_pct behavior is used when 'mode' key is absent from risk config."""
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
    trades = run_strategy_on_dataframe(df, strategy, initial_balance=10000.0)
    assert isinstance(trades, list)
    for trade in trades:
        assert "entry_price" in trade
        assert "position_size" in trade


def test_atr_mode_produces_different_sizing_than_fixed_pct() -> None:
    """When mode == 'atr', position sizing uses ATR-based calculation, not position_size_pct."""
    base = {
        "version": "0001",
        "asset": "BTC/USDT",
        "timeframe": "1h",
        "entry": {"indicator": "rsi", "threshold": 80, "direction": "long"},
        "exit": {"rsi_take_profit": 95},
        "costs": {"fee_pct": 0.1, "slippage_pct": 0.05},
        "reflection": {"one_variable_only": True, "allowed_variables": []},
    }
    df = generate_sample_ohlcv(rows=50, seed=42)

    strategy_fixed = {**base, "risk": {"stop_loss_pct": 2.0, "position_size_pct": 10.0}}
    trades_fixed = run_strategy_on_dataframe(df, strategy_fixed, initial_balance=10000.0)

    strategy_atr = {
        **base,
        "risk": {"mode": "atr", "risk_pct": 1.0, "atr_period": 2, "atr_stop_multiplier": 2.0},
    }
    trades_atr = run_strategy_on_dataframe(df, strategy_atr, initial_balance=10000.0)

    assert isinstance(trades_atr, list)
    # Both modes should produce at least one trade with the lenient RSI threshold.
    assert len(trades_fixed) > 0
    assert len(trades_atr) > 0
    # ATR-based sizing produces different position sizes than fixed 10%.
    fixed_sizes = [round(t["position_size"], 8) for t in trades_fixed]
    atr_sizes = [round(t["position_size"], 8) for t in trades_atr]
    assert fixed_sizes != atr_sizes
