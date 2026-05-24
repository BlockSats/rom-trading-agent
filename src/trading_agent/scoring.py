from __future__ import annotations

from math import sqrt, tanh
from typing import Any


def _returns_fraction(trades: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for trade in trades:
        values.append(float(trade.get("return_pct", 0.0)) / 100.0)
    return values


def calculate_total_return(trades: list[dict[str, Any]]) -> float:
    total = 0.0
    for trade in trades:
        total += float(trade.get("return_pct", 0.0)) / 100.0
    return total


def calculate_max_drawdown(trades: list[dict[str, Any]]) -> float:
    if not trades:
        return 0.0

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for trade_return in _returns_fraction(trades):
        equity *= 1.0 + trade_return
        peak = max(peak, equity)
        if peak > 0:
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def calculate_sharpe_like(trades: list[dict[str, Any]]) -> float:
    returns = _returns_fraction(trades)
    if len(returns) < 2:
        return sum(returns) * 10.0

    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1)
    if variance <= 0:
        return mean_return * 10.0
    return mean_return / sqrt(variance) * sqrt(len(returns))


def score_trades(trades: list[dict[str, Any]], goal: dict[str, Any]) -> dict[str, float]:
    total_return = calculate_total_return(trades)
    max_drawdown = calculate_max_drawdown(trades)
    sharpe_like = calculate_sharpe_like(trades)

    if not trades:
        score = 0.0
    else:
        drawdown_penalty = max(0.0, max_drawdown - float(goal.get("max_drawdown", 0.0))) * 5.0
        sharpe_penalty = max(0.0, float(goal.get("min_sharpe", 0.0)) - sharpe_like) * 0.2
        raw_score = total_return * 4.0 - drawdown_penalty - sharpe_penalty
        score = tanh(raw_score)

    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "sharpe_like": float(sharpe_like),
        "score": float(max(min(score, 1.0), -1.0)),
    }

