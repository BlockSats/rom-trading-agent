from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from trading_agent.backtest import run_backtest, summarize_backtest
from trading_agent.scoring import score_trades
from trading_agent.storage import write_json
from trading_agent.walk_forward import split_walk_forward_windows


def build_walk_forward_report(
    df: pd.DataFrame,
    strategy: dict[str, Any],
    goal: dict[str, Any],
    train_window: int,
    test_window: int,
    step: int,
) -> dict[str, Any]:
    """Run a descriptive walk-forward analysis on a single strategy.

    Backtests are run on the test window only. The train window is kept as
    temporal context in the report but is never used for parameter optimisation.
    Nothing is written to disk. No network calls are made.

    Parameters
    ----------
    df:
        Chronological OHLCV DataFrame with a ``timestamp`` column.
    strategy:
        Strategy configuration dict.
    goal:
        Scoring goal dict with at least ``max_drawdown`` and ``min_sharpe``.
    train_window:
        Number of rows in each in-sample slice.
    test_window:
        Number of rows in each out-of-sample slice.
    step:
        Number of rows to advance between windows.

    Returns
    -------
    dict
        Descriptive report with per-window results and a global summary.
        Never contains ranking, selection, or promotion fields.
    """
    windows = split_walk_forward_windows(df, train_window, test_window, step)

    results: list[dict[str, Any]] = []
    for w in windows:
        backtest_result = run_backtest(w.test, strategy)
        trades = backtest_result["trades"]
        score_result = score_trades(trades, goal)
        summary = summarize_backtest(trades, score_result)
        summary["winrate"] = score_result["winrate"]
        summary["profit_factor"] = score_result["profit_factor"]
        summary["expectancy"] = score_result["expectancy"]

        results.append(
            {
                "window_index": w.index,
                "train_start": str(w.train_start),
                "train_end": str(w.train_end),
                "test_start": str(w.test_start),
                "test_end": str(w.test_end),
                "test_rows": len(w.test),
                "train_role": "historical_context",
                "test_role": "walk_forward_oos",
                "test_is_out_of_sample": True,
                "test_is_confirmatory_holdout": False,
                "test_is_paper_forward": False,
                "train_used_for_parameter_optimization": False,
                "summary": summary,
            }
        )

    total_trades = sum(r["summary"]["closed_trades"] for r in results)
    windows_with_trades = sum(1 for r in results if r["summary"]["closed_trades"] > 0)

    return {
        "walk_forward": {
            "train_window": train_window,
            "test_window": test_window,
            "step": step,
            "windows": len(windows),
        },
        "strategy": {
            "strategy_id": strategy.get("strategy_id", strategy.get("entry", {}).get("indicator", "")),
            "asset": strategy.get("asset", ""),
            "timeframe": strategy.get("timeframe", ""),
        },
        "validation_context": {
            "mode": "exploratory_walk_forward",
            "data_role": "exploratory_data",
            "confirmatory_holdout_used": False,
            "prospective_holdout_used": False,
            "paper_forward_used": False,
            "parameter_optimization_performed": False,
            "selection_performed": False,
        },
        "results": results,
        "summary": {
            "windows": len(windows),
            "windows_with_trades": windows_with_trades,
            "total_trades": total_trades,
        },
    }


def save_walk_forward_report(
    report: dict[str, Any],
    output_path: Path | str = Path("outputs/walk_forward_report.json"),
) -> Path:
    path = Path(output_path)
    write_json(path, report)
    return path
