from __future__ import annotations

from typing import Any

import pandas as pd

from trading_agent.config import get_active_strategy_id
from trading_agent.indicators import atr, ema, rsi


def evaluate_rsi_signal(latest_rsi: float, strategy: dict[str, Any]) -> str:
    entry_threshold = float(strategy["entry"]["threshold"])
    exit_threshold = float(strategy["exit"]["rsi_take_profit"])
    if latest_rsi <= entry_threshold:
        return "buy"
    if latest_rsi >= exit_threshold:
        return "sell"
    return "hold"


def generate_rsi_signals(df: pd.DataFrame, strategy: dict[str, Any]) -> pd.Series:
    if "close" not in df.columns:
        raise ValueError("dataframe must contain a close column")

    closes = df["close"].astype(float).tolist()
    indicator = rsi(closes)
    signals = pd.Series("hold", index=df.index, dtype="object")

    for position, latest_rsi in enumerate(indicator):
        if pd.isna(latest_rsi):
            continue
        signals.iloc[position] = evaluate_rsi_signal(float(latest_rsi), strategy)

    return signals


def generate_ema_atr_trend_signals(df: pd.DataFrame, strategy: dict[str, Any]) -> pd.Series:
    signals = pd.Series("hold", index=df.index, dtype="object")
    required_columns = {"high", "low", "close"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"dataframe must contain columns: {', '.join(missing_columns)}")

    fast_ema_period = int(strategy["entry"]["fast_ema_period"])
    slow_ema_period = int(strategy["entry"]["slow_ema_period"])
    atr_period = int(strategy["exit"]["atr_period"])
    if fast_ema_period <= 0 or slow_ema_period <= 0 or atr_period <= 0:
        raise ValueError("EMA and ATR periods must be positive")

    minimum_required_rows = max(slow_ema_period, atr_period + 1)
    if len(df) < minimum_required_rows:
        return signals

    closes = df["close"].astype(float)
    fast_ema = ema(closes.tolist(), fast_ema_period)
    slow_ema = ema(closes.tolist(), slow_ema_period)
    atr_values = atr(
        df["high"].astype(float).tolist(),
        df["low"].astype(float).tolist(),
        closes.tolist(),
        atr_period,
    )
    atr_multiplier = float(strategy["exit"]["atr_stop_multiplier"])

    in_position = False
    highest_close = 0.0

    for position in range(len(df)):
        latest_fast = fast_ema.iloc[position]
        latest_slow = slow_ema.iloc[position]
        latest_atr = atr_values.iloc[position]
        close_price = float(closes.iloc[position])

        if pd.isna(latest_fast) or pd.isna(latest_slow) or pd.isna(latest_atr):
            continue

        previous_fast = fast_ema.iloc[position - 1] if position > 0 else pd.NA
        previous_slow = slow_ema.iloc[position - 1] if position > 0 else pd.NA
        if pd.isna(previous_fast) or pd.isna(previous_slow):
            continue

        bullish_cross = float(previous_fast) <= float(previous_slow) and float(latest_fast) > float(latest_slow)
        bearish_cross = float(previous_fast) >= float(previous_slow) and float(latest_fast) < float(latest_slow)

        if not in_position and bullish_cross:
            signals.iloc[position] = "buy"
            in_position = True
            highest_close = close_price
            continue

        if in_position:
            highest_close = max(highest_close, close_price)
            atr_stop = highest_close - (float(latest_atr) * atr_multiplier)
            if bearish_cross or close_price <= atr_stop:
                signals.iloc[position] = "sell"
                in_position = False
                highest_close = 0.0

    return signals


def generate_donchian_breakout_signals(df: pd.DataFrame, strategy: dict[str, Any]) -> pd.Series:
    signals = pd.Series("hold", index=df.index, dtype="object")
    required_columns = {"high", "low", "close"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"dataframe must contain columns: {', '.join(missing_columns)}")

    donchian_period = int(strategy["entry"]["donchian_period"])
    atr_period = int(strategy["exit"]["atr_period"])
    atr_stop_multiplier = float(strategy["exit"]["atr_stop_multiplier"])

    # Guard: atr() raises if fewer than atr_period + 1 rows; also need donchian_period + 1 for shift.
    minimum_required_rows = max(donchian_period + 1, atr_period + 1)
    if len(df) < minimum_required_rows:
        return signals

    closes = df["close"].astype(float)
    highs = df["high"].astype(float)

    # shift(1) prevents the current candle's high from contributing to its own breakout level.
    breakout_level = highs.rolling(donchian_period).max().shift(1)

    atr_values = atr(
        highs.tolist(),
        df["low"].astype(float).tolist(),
        closes.tolist(),
        atr_period,
    )

    in_position = False
    highest_close = 0.0

    for position in range(len(df)):
        close_price = float(closes.iloc[position])
        level = breakout_level.iloc[position]
        latest_atr = atr_values.iloc[position]

        if pd.isna(level) or pd.isna(latest_atr):
            continue

        if not in_position:
            if close_price > float(level):
                signals.iloc[position] = "buy"
                in_position = True
                highest_close = close_price
        else:
            highest_close = max(highest_close, close_price)
            atr_stop = highest_close - float(latest_atr) * atr_stop_multiplier
            if close_price <= atr_stop:
                signals.iloc[position] = "sell"
                in_position = False
                highest_close = 0.0

    return signals


def generate_signals(df: pd.DataFrame, strategy: dict[str, Any]) -> pd.Series:
    strategy_id = get_active_strategy_id(strategy)
    if strategy_id == "ema_atr_trend":
        return generate_ema_atr_trend_signals(df, strategy)
    if strategy_id == "donchian_breakout":
        return generate_donchian_breakout_signals(df, strategy)
    return generate_rsi_signals(df, strategy)
