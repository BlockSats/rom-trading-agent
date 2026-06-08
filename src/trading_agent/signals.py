from __future__ import annotations

from typing import Any

import pandas as pd

from trading_agent.indicators import rsi


def evaluate_rsi_signal(latest_rsi: float, strategy: dict[str, Any]) -> str:
    entry_threshold = float(strategy["entry"]["threshold"])
    exit_threshold = float(strategy["exit"]["rsi_take_profit"])
    if latest_rsi <= entry_threshold:
        return "buy"
    if latest_rsi >= exit_threshold:
        return "sell"
    return "hold"


def generate_signals(df: pd.DataFrame, strategy: dict[str, Any]) -> pd.Series:
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
