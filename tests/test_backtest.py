from __future__ import annotations

import pandas as pd

from trading_agent.backtest import run_backtest, summarize_backtest
from trading_agent.data import generate_sample_ohlcv
from trading_agent.scoring import score_trades


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


def test_run_backtest_returns_expected_keys() -> None:
    df = generate_sample_ohlcv(rows=50, seed=42)
    result = run_backtest(df, _strategy())
    assert {"initial_balance", "closed_trades", "total_net_pnl", "trades"} <= result.keys()
    assert isinstance(result["trades"], list)


def test_run_backtest_returns_a_list_of_trades() -> None:
    df = generate_sample_ohlcv(rows=50, seed=42)
    result = run_backtest(df, _strategy())
    assert isinstance(result["trades"], list)


def test_summarize_backtest_returns_scoring_fields() -> None:
    df = generate_sample_ohlcv(rows=50, seed=42)
    trades = run_backtest(df, _strategy())["trades"]
    score_result = score_trades(trades, {"max_drawdown": 0.08, "min_sharpe": 1.2})
    summary = summarize_backtest(trades, score_result)
    assert {"closed_trades", "total_net_pnl", "total_return", "max_drawdown", "sharpe_like", "score"} <= summary.keys()


def test_empty_backtest_does_not_crash() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="h"),
            "open": [100.0] * 20,
            "high": [101.0] * 20,
            "low": [99.0] * 20,
            "close": list(range(100, 120)),
            "volume": [10.0] * 20,
        }
    )
    result = run_backtest(df, _strategy())
    score_result = score_trades(result["trades"], {"max_drawdown": 0.08, "min_sharpe": 1.2})
    summary = summarize_backtest(result["trades"], score_result)
    assert result["closed_trades"] == 0
    assert summary["closed_trades"] == 0
