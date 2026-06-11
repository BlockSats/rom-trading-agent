from __future__ import annotations

import pandas as pd

from trading_agent.signals import generate_donchian_breakout_signals, generate_signals


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


def _ema_atr_strategy() -> dict:
    return {
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


def _frame(closes: list[float]) -> pd.DataFrame:
    index = pd.Index(range(100, 100 + len(closes)))
    return pd.DataFrame({"close": closes}, index=index)


def _ohlc_frame(closes: list[float]) -> pd.DataFrame:
    index = pd.Index(range(100, 100 + len(closes)))
    return pd.DataFrame(
        {
            "high": [price + 1.0 for price in closes],
            "low": [price - 1.0 for price in closes],
            "close": closes,
        },
        index=index,
    )


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


def test_generate_signals_supports_ema_atr_trend_buy_and_sell() -> None:
    df = _ohlc_frame([10.0, 10.0, 10.0, 10.0, 12.0, 14.0, 16.0, 18.0, 15.0, 13.0, 11.0])

    signals = generate_signals(df, _ema_atr_strategy())

    assert "buy" in set(signals)
    assert "sell" in set(signals)


def test_generate_signals_ema_atr_requires_ohlc_columns() -> None:
    df = _frame([10.0, 10.0, 12.0, 14.0, 16.0])

    try:
        generate_signals(df, _ema_atr_strategy())
    except ValueError as exc:
        assert "dataframe must contain columns" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_generate_signals_ema_atr_short_history_returns_hold() -> None:
    df = _ohlc_frame([10.0, 11.0, 12.0])

    signals = generate_signals(df, _ema_atr_strategy())

    assert len(signals) == len(df)
    assert signals.index.equals(df.index)
    assert set(signals) == {"hold"}


# --- Donchian Breakout signals ---


def _donchian_strategy() -> dict:
    return {
        "version": "0001",
        "strategy_id": "donchian_breakout",
        "asset": "BTC/USDT",
        "timeframe": "1h",
        "entry": {
            "indicator": "donchian",
            "donchian_period": 3,
            "direction": "long",
        },
        "exit": {"atr_period": 2, "atr_stop_multiplier": 1.5},
        "risk": {"stop_loss_pct": 2.0, "position_size_pct": 10.0},
        "costs": {"fee_pct": 0.1, "slippage_pct": 0.05},
        "reflection": {"one_variable_only": True, "allowed_variables": ["entry.donchian_period"]},
    }


def _donchian_frame(highs: list[float], lows: list[float], closes: list[float]) -> pd.DataFrame:
    index = pd.Index(range(len(closes)))
    return pd.DataFrame({"high": highs, "low": lows, "close": closes}, index=index)


def test_donchian_breakout_buy_on_channel_breakout() -> None:
    # donchian_period=3, atr_period=2 → minimum 4 rows
    # After 4 rows, breakout_level = max(highs[0:3]) = 10
    # close[3] = 15 > 10 → buy
    highs =  [10.0, 10.0, 10.0, 12.0]
    lows =   [5.0,  5.0,  5.0,  5.0]
    closes = [9.0,  9.0,  9.0,  15.0]
    df = _donchian_frame(highs, lows, closes)

    signals = generate_donchian_breakout_signals(df, _donchian_strategy())

    assert "buy" in set(signals)
    assert signals.iloc[3] == "buy"


def test_donchian_breakout_hold_during_donchian_warmup() -> None:
    # Only 2 rows: less than max(donchian_period+1=4, atr_period+1=3) = 4 → all hold
    df = _donchian_frame([10.0, 12.0], [5.0, 5.0], [9.0, 15.0])

    signals = generate_donchian_breakout_signals(df, _donchian_strategy())

    assert set(signals) == {"hold"}


def test_donchian_breakout_hold_during_atr_warmup() -> None:
    # donchian_period=3, atr_period=2 → minimum 4 rows
    # At position 2: breakout level valid (shift 1, rolling 3 needs 3 rows = positions 0-2)
    # but ATR(period=2) needs 3 rows; with ewm min_periods=2, first valid ATR at position 2
    # Position 3 has both valid → buy or sell possible; positions 0-2 should hold
    highs =  [10.0, 10.0, 10.0, 12.0]
    lows =   [5.0,  5.0,  5.0,  5.0]
    closes = [9.0,  9.0,  9.0,  9.0]
    df = _donchian_frame(highs, lows, closes)

    signals = generate_donchian_breakout_signals(df, _donchian_strategy())

    # Positions 0-2 must all be hold (breakout_level NaN or ATR NaN)
    assert signals.iloc[0] == "hold"
    assert signals.iloc[1] == "hold"
    assert signals.iloc[2] == "hold"


def test_donchian_breakout_sell_on_atr_trailing_stop() -> None:
    # donchian_period=3, atr_period=2, atr_stop_multiplier=1.5
    # Tight H-L ranges (0.2) keep ATR small; close[7]=14.0 falls below the trailing stop.
    # stop at pos 7 = 15.0 - 0.353 * 1.5 ≈ 14.47 → sell triggers.
    highs =  [10.1, 10.1, 10.1, 15.1, 15.1, 15.1, 15.1, 15.1]
    lows =   [9.9,  9.9,  9.9,  14.9, 14.9, 14.9, 14.9, 14.9]
    closes = [10.0, 10.0, 10.0, 15.0, 15.0, 15.0, 15.0, 14.0]
    df = _donchian_frame(highs, lows, closes)

    signals = generate_donchian_breakout_signals(df, _donchian_strategy())

    assert "buy" in set(signals)
    assert "sell" in set(signals)
    # sell must come after buy
    buy_idx = list(signals).index("buy")
    sell_idx = list(signals).index("sell")
    assert sell_idx > buy_idx


def test_donchian_no_lookahead_from_current_candle_high() -> None:
    # high[3]=25 is a new record but close[3]=15 < breakout_level = max(highs[0:3]) = 20
    # With shift(1), current candle's high does not raise the breakout threshold.
    # → position 3 must be hold even though high is a record.
    highs =  [10.0, 15.0, 20.0, 25.0]
    lows =   [5.0,  5.0,  5.0,  5.0]
    closes = [9.0,  14.0, 19.0, 15.0]
    df = _donchian_frame(highs, lows, closes)

    signals = generate_donchian_breakout_signals(df, _donchian_strategy())

    # close[3]=15 < breakout_level=max(highs[0:3])=20 → no buy
    assert signals.iloc[3] == "hold"
