from __future__ import annotations

from typing import Any

import pandas as pd

from trading_agent.scoring import calculate_max_drawdown, calculate_sharpe_like, calculate_total_return
from trading_agent.strategy import run_strategy_on_dataframe


def run_backtest(
    df: pd.DataFrame,
    strategy: dict[str, Any],
    initial_balance: float = 10000.0,
) -> dict[str, Any]:
    trades = run_strategy_on_dataframe(df, strategy, initial_balance=initial_balance)
    total_net_pnl = sum(float(trade.get("net_pnl", 0.0)) for trade in trades)
    return {
        "initial_balance": float(initial_balance),
        "closed_trades": len(trades),
        "total_net_pnl": float(total_net_pnl),
        "trades": trades,
    }


def summarize_backtest(trades: list[dict[str, Any]], score_result: dict[str, Any]) -> dict[str, Any]:
    total_net_pnl = sum(float(trade.get("net_pnl", 0.0)) for trade in trades)
    return {
        "closed_trades": len(trades),
        "total_net_pnl": float(total_net_pnl),
        "total_return": float(calculate_total_return(trades)),
        "max_drawdown": float(calculate_max_drawdown(trades)),
        "sharpe_like": float(calculate_sharpe_like(trades)),
        "score": float(score_result.get("score", 0.0)),
    }
