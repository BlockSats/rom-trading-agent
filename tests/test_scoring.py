from __future__ import annotations

from trading_agent.scoring import score_trades


def test_empty_trades_do_not_crash() -> None:
    result = score_trades([], {"max_drawdown": 0.08, "min_sharpe": 1.2})
    assert result["score"] == 0.0
    assert result["winrate"] == 0.0
    assert result["profit_factor"] is None
    assert result["expectancy"] == 0.0


def test_score_is_always_between_minus_one_and_plus_one() -> None:
    result = score_trades(
        [{"return_pct": 1.0}, {"return_pct": -0.5}],
        {"max_drawdown": 0.08, "min_sharpe": 1.2},
    )
    assert -1.0 <= result["score"] <= 1.0


def test_positive_trades_produce_better_score_than_negative() -> None:
    goal = {"max_drawdown": 0.08, "min_sharpe": 1.2}
    positive = score_trades([{"return_pct": 2.0}, {"return_pct": 1.0}], goal)
    negative = score_trades([{"return_pct": -2.0}, {"return_pct": -1.0}], goal)
    assert positive["score"] > negative["score"]


def test_trade_quality_metrics_with_winners_and_losers() -> None:
    trades = [
        {"return_pct": 1.0, "net_pnl": 100.0},
        {"return_pct": -0.5, "net_pnl": -50.0},
        {"return_pct": 2.0, "net_pnl": 150.0},
        {"return_pct": -1.0, "net_pnl": -25.0},
    ]

    result = score_trades(trades, {"max_drawdown": 0.08, "min_sharpe": 1.2})

    assert result["winrate"] == 0.5
    assert result["profit_factor"] == 250.0 / 75.0
    assert result["expectancy"] == 43.75


def test_trade_quality_metrics_with_only_winners() -> None:
    trades = [
        {"return_pct": 1.0, "net_pnl": 100.0},
        {"return_pct": 2.0, "net_pnl": 50.0},
    ]

    result = score_trades(trades, {"max_drawdown": 0.08, "min_sharpe": 1.2})

    assert result["winrate"] == 1.0
    assert result["profit_factor"] is None
    assert result["expectancy"] == 75.0


def test_trade_quality_metrics_with_only_losers() -> None:
    trades = [
        {"return_pct": -1.0, "net_pnl": -100.0},
        {"return_pct": -2.0, "net_pnl": -50.0},
    ]

    result = score_trades(trades, {"max_drawdown": 0.08, "min_sharpe": 1.2})

    assert result["winrate"] == 0.0
    assert result["profit_factor"] == 0.0
    assert result["expectancy"] == -75.0
