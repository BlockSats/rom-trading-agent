from __future__ import annotations

import math
from typing import Any

import pandas as pd

from trading_agent.backtest import run_backtest, summarize_backtest
from trading_agent.reflection import apply_reflection_proposal
from trading_agent.research_reflection import decide_reflection
from trading_agent.scoring import score_trades


def split_ohlcv_windows(ohlcv: pd.DataFrame, windows: int) -> list[pd.DataFrame]:
    """Split OHLCV data into chronological validation windows."""
    if windows < 3:
        raise ValueError("Robustness validation requires at least 3 windows.")

    if len(ohlcv) < windows:
        raise ValueError("Not enough OHLCV rows for the requested number of windows.")

    window_size = len(ohlcv) // windows

    if window_size <= 0:
        raise ValueError("Window size must be greater than zero.")

    result: list[pd.DataFrame] = []

    for index in range(windows):
        start = index * window_size
        end = (index + 1) * window_size if index < windows - 1 else len(ohlcv)
        result.append(ohlcv.iloc[start:end].copy())

    return result


def summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate robustness metrics from per-window comparisons."""
    kept = [report for report in reports if report["decision"] == "keep"]
    rejected = [report for report in reports if report["decision"] == "reject"]

    def avg(key: str) -> float:
        if not reports:
            return 0.0
        return sum(float(report[key]) for report in reports) / len(reports)

    return {
        "kept_windows": len(kept),
        "rejected_windows": len(rejected),
        "average_baseline_score": avg("baseline_score"),
        "average_candidate_score": avg("candidate_score"),
        "average_baseline_drawdown": avg("baseline_drawdown"),
        "average_candidate_drawdown": avg("candidate_drawdown"),
        "average_baseline_closed_trades": avg("baseline_closed_trades"),
        "average_candidate_closed_trades": avg("candidate_closed_trades"),
    }


def decide_robustness(
    *,
    windows_results: list[dict[str, Any]],
    min_keep_ratio: float = 0.75,
    min_average_candidate_trades: float = 1.0,
) -> tuple[str, str]:
    """Decide whether a hypothesis is robust enough across windows."""
    summary = summarize_reports(windows_results)

    required_kept_windows = math.ceil(len(windows_results) * min_keep_ratio)

    if summary["kept_windows"] < required_kept_windows:
        return "reject", "Candidate did not improve across enough windows."

    if summary["average_candidate_drawdown"] > summary["average_baseline_drawdown"]:
        return "reject", "Candidate increased average drawdown."

    if summary["average_candidate_closed_trades"] < min_average_candidate_trades:
        return "reject", "Candidate produced too few trades on average."

    if summary["average_candidate_score"] <= summary["average_baseline_score"]:
        return "reject", "Candidate did not improve average score."

    return (
        "keep",
        "Candidate improved score across enough windows without increasing average drawdown.",
    )


def build_research_robustness_report(
    *,
    symbol: str,
    interval: str,
    limit: int,
    windows: int,
    ohlcv: pd.DataFrame,
    strategy: dict[str, Any],
    goal: dict[str, Any],
    proposal: dict[str, Any],
) -> dict[str, Any]:
    """Compare baseline vs reflected candidate across multiple OHLCV windows."""
    candidate_strategy = apply_reflection_proposal(strategy, proposal)
    ohlcv_windows = split_ohlcv_windows(ohlcv, windows)

    windows_results: list[dict[str, Any]] = []

    for index, window in enumerate(ohlcv_windows, start=1):
        baseline_backtest = run_backtest(window, strategy)
        baseline_score = score_trades(baseline_backtest["trades"], goal)
        baseline_summary = summarize_backtest(
            baseline_backtest["trades"],
            baseline_score,
        )

        candidate_backtest = run_backtest(window, candidate_strategy)
        candidate_score = score_trades(candidate_backtest["trades"], goal)
        candidate_summary = summarize_backtest(
            candidate_backtest["trades"],
            candidate_score,
        )

        decision, decision_reason = decide_reflection(
            baseline_report=baseline_summary,
            candidate_report=candidate_summary,
        )

        windows_results.append(
            {
                "window": index,
                "rows": int(len(window)),
                "baseline_score": baseline_summary["score"],
                "candidate_score": candidate_summary["score"],
                "baseline_drawdown": baseline_summary["max_drawdown"],
                "candidate_drawdown": candidate_summary["max_drawdown"],
                "baseline_closed_trades": baseline_summary["closed_trades"],
                "candidate_closed_trades": candidate_summary["closed_trades"],
                "decision": decision,
                "decision_reason": decision_reason,
            }
        )

    summary = summarize_reports(windows_results)
    final_decision, decision_reason = decide_robustness(
        windows_results=windows_results,
    )

    return {
        "command": "research-robustness",
        "status": "completed",
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
        "windows": windows,
        "hypothesis": {
            "variable": proposal["variable"],
            "old_value": proposal["old_value"],
            "new_value": proposal["new_value"],
            "reason": proposal.get("reason"),
            "expected_effect": proposal.get("expected_effect"),
            "confidence": proposal.get("confidence"),
        },
        "windows_results": windows_results,
        "summary": summary,
        "final_decision": final_decision,
        "decision_reason": decision_reason,
    }
