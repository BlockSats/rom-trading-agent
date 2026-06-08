from __future__ import annotations

import pandas as pd

from trading_agent.signals import generate_signals


def _strategy() -> dict:
    return {
        "version": "0001",
        "asset": "BTC/USDT",
        "timeframe": "1h",
        "entry": {"indicator": "rsi", "threshold": 30, "direction": "long"},
        "exit": {"rsi_take_profit": 55},
        "risk": {"stop_loss_pct": 2.0, "position_size_pct": 10.0},
        "costs": {"fee_pct": 0.1, "slippage_pct": 0.05},
        "reflection": {"one_variable_only": True, "allowed_variables": ["entry.threshold"]},
    }


def _frame(closes: list[float]) -> pd.DataFrame:
    index = pd.Index(range(100, 100 + len(closes)))
    return pd.DataFrame({"close": closes}, index=index)


def test_generate_signals_returns_buy_when_rsi_entry_triggers() -> None:
    df = _frame([120.0 - float(index) for index in range(20)])

    signals = generate_signals(df, _strategy())

    assert "buy" in set(signals)


def test_generate_signals_returns_sell_when_rsi_exit_triggers() -> None:
    df = _frame([100.0 + float(index) for index in range(20)])

    signals = generate_signals(df, _strategy())

    assert "sell" in set(signals)


def test_generate_signals_returns_hold_when_no_condition_triggers() -> None:
    df = _frame([100.0] * 20)

    signals = generate_signals(df, _strategy())

    assert set(signals) == {"hold"}


def test_generate_signals_returns_hold_while_rsi_is_unavailable() -> None:
    df = _frame([120.0 - float(index) for index in range(20)])

    signals = generate_signals(df, _strategy())

    assert signals.iloc[:14].tolist() == ["hold"] * 14


def test_generate_signals_keeps_dataframe_length_and_index() -> None:
    df = _frame([100.0] * 20)

    signals = generate_signals(df, _strategy())

    assert len(signals) == len(df)
    assert signals.index.equals(df.index)
