from __future__ import annotations

from typing import Any

import pandas as pd

from trading_agent.broker_paper import PaperBroker
from trading_agent.indicators import rsi


def _serialize_timestamp(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def evaluate_rsi_signal(latest_rsi: float, strategy: dict[str, Any]) -> str:
    entry_threshold = float(strategy["entry"]["threshold"])
    exit_threshold = float(strategy["exit"]["rsi_take_profit"])
    if latest_rsi <= entry_threshold:
        return "buy"
    if latest_rsi >= exit_threshold:
        return "sell"
    return "hold"


def run_strategy_on_dataframe(
    df: pd.DataFrame,
    strategy: dict[str, Any],
    initial_balance: float = 10000.0,
) -> list[dict[str, Any]]:
    if "close" not in df.columns:
        raise ValueError("dataframe must contain a close column")

    closes = df["close"].astype(float).tolist()
    indicator = rsi(closes)

    broker = PaperBroker(initial_balance=initial_balance)
    closed_trades: list[dict[str, Any]] = []
    last_timestamp = None
    last_close = None

    for index, row in df.reset_index(drop=True).iterrows():
        latest_rsi = indicator.iloc[index]
        if pd.isna(latest_rsi):
            continue

        close_price = float(row["close"])
        timestamp = row["timestamp"] if "timestamp" in df.columns else None
        last_timestamp = timestamp
        last_close = close_price
        stop_loss_pct = float(strategy["risk"]["stop_loss_pct"])
        fee_pct = float(strategy["costs"]["fee_pct"])
        slippage_pct = float(strategy["costs"]["slippage_pct"])

        if broker.is_open:
            entry_price = float(broker.entry_price or close_price)
            stop_loss_price = entry_price * (1 - stop_loss_pct / 100.0)
            signal = evaluate_rsi_signal(float(latest_rsi), strategy)
            if close_price <= stop_loss_price or signal == "sell":
                trade = broker.sell(close_price, fee_pct=fee_pct, slippage_pct=slippage_pct)
                trade["reason"] = "stop_loss" if close_price <= stop_loss_price else "rsi_take_profit"
                if timestamp is not None:
                    trade["exit_timestamp"] = _serialize_timestamp(timestamp)
                closed_trades.append(trade)
        else:
            signal = evaluate_rsi_signal(float(latest_rsi), strategy)
            if signal == "buy":
                broker.buy(close_price, float(strategy["risk"]["position_size_pct"]), fee_pct=fee_pct, slippage_pct=slippage_pct)
                if timestamp is not None:
                    broker._entry_timestamp = timestamp  # type: ignore[attr-defined]

    if broker.is_open and last_close is not None:
        trade = broker.sell(last_close, fee_pct=fee_pct, slippage_pct=slippage_pct)
        trade["reason"] = "end_of_data"
        if last_timestamp is not None:
            trade["exit_timestamp"] = _serialize_timestamp(last_timestamp)
        closed_trades.append(trade)

    return closed_trades
