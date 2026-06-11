from __future__ import annotations

import math
from typing import Any

import pandas as pd

from trading_agent.broker_paper import PaperBroker
from trading_agent.config import get_active_strategy_id
from trading_agent.indicators import atr as _compute_atr
from trading_agent.risk_model import compute_atr_position_size, compute_atr_stop_price
from trading_agent.signals import generate_signals


def _serialize_timestamp(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def run_strategy_on_dataframe(
    df: pd.DataFrame,
    strategy: dict[str, Any],
    initial_balance: float = 10000.0,
) -> list[dict[str, Any]]:
    if "close" not in df.columns:
        raise ValueError("dataframe must contain a close column")

    signals = generate_signals(df, strategy)
    signal_exit_reason = "rsi_take_profit" if get_active_strategy_id(strategy) == "rsi_baseline" else "strategy_exit"

    risk_mode = strategy["risk"].get("mode", "fixed_pct")

    # Pre-compute ATR series for ATR mode (requires high/low/close columns).
    atr_series: pd.Series | None = None
    if risk_mode == "atr":
        atr_period = int(strategy["risk"].get("atr_period", 14))
        required_cols = {"high", "low", "close"}
        if required_cols.issubset(df.columns) and len(df) >= atr_period + 1:
            atr_series = _compute_atr(
                df["high"].astype(float).tolist(),
                df["low"].astype(float).tolist(),
                df["close"].astype(float).tolist(),
                atr_period,
            )

    broker = PaperBroker(initial_balance=initial_balance)
    closed_trades: list[dict[str, Any]] = []
    last_timestamp = None
    last_close = None
    fee_pct = float(strategy["costs"]["fee_pct"])
    slippage_pct = float(strategy["costs"]["slippage_pct"])
    atr_stop_price: float | None = None

    for index, row in df.reset_index(drop=True).iterrows():
        signal = str(signals.iloc[index])
        close_price = float(row["close"])
        timestamp = row["timestamp"] if "timestamp" in df.columns else None
        last_timestamp = timestamp
        last_close = close_price

        if broker.is_open:
            entry_price = float(broker.entry_price or close_price)
            if risk_mode == "atr" and atr_stop_price is not None:
                stop_loss_price = atr_stop_price
            else:
                stop_loss_pct = float(strategy["risk"]["stop_loss_pct"])
                stop_loss_price = entry_price * (1 - stop_loss_pct / 100.0)

            if close_price <= stop_loss_price or signal == "sell":
                trade = broker.sell(close_price, fee_pct=fee_pct, slippage_pct=slippage_pct)
                trade["reason"] = "stop_loss" if close_price <= stop_loss_price else signal_exit_reason
                if timestamp is not None:
                    trade["exit_timestamp"] = _serialize_timestamp(timestamp)
                closed_trades.append(trade)
                atr_stop_price = None
        else:
            if signal == "buy":
                if risk_mode == "atr":
                    atr_val = float(atr_series.iloc[index]) if atr_series is not None else float("nan")
                    if math.isnan(atr_val) or atr_val <= 0:
                        continue
                    risk_pct = float(strategy["risk"]["risk_pct"])
                    atr_mult = float(strategy["risk"]["atr_stop_multiplier"])
                    size_pct = compute_atr_position_size(broker.cash, close_price, atr_val, risk_pct, atr_mult)
                    atr_stop_price = compute_atr_stop_price(close_price, atr_val, atr_mult)
                else:
                    size_pct = float(strategy["risk"]["position_size_pct"])
                    atr_stop_price = None

                broker.buy(close_price, size_pct, fee_pct=fee_pct, slippage_pct=slippage_pct)
                if timestamp is not None:
                    broker._entry_timestamp = timestamp  # type: ignore[attr-defined]

    if broker.is_open and last_close is not None:
        trade = broker.sell(last_close, fee_pct=fee_pct, slippage_pct=slippage_pct)
        trade["reason"] = "end_of_data"
        if last_timestamp is not None:
            trade["exit_timestamp"] = _serialize_timestamp(last_timestamp)
        closed_trades.append(trade)

    return closed_trades
