from __future__ import annotations

import pandas as pd
import pytest

from trading_agent.research_robustness import (
    decide_robustness,
    split_ohlcv_windows,
    summarize_reports,
)


def test_split_ohlcv_windows_requires_at_least_three_windows() -> None:
    ohlcv = pd.DataFrame({"close": [1, 2, 3, 4]})

    with pytest.raises(ValueError, match="at least 3 windows"):
        split_ohlcv_windows(ohlcv, 2)


def test_split_ohlcv_windows_preserves_all_rows() -> None:
    ohlcv = pd.DataFrame({"close": list(range(10))})

    windows = split_ohlcv_windows(ohlcv, 4)

    assert len(windows) == 4
    assert sum(len(window) for window in windows) == 10
    assert windows[0]["close"].iloc[0] == 0
    assert windows[-1]["close"].iloc[-1] == 9


def test_summarize_reports_counts_keep_and_reject() -> None:
    reports = [
        {
            "decision": "keep",
            "baseline_score": -0.8,
            "candidate_score": -0.7,
            "baseline_drawdown": 0.2,
            "candidate_drawdown": 0.18,
            "baseline_closed_trades": 4,
            "candidate_closed_trades": 4,
        },
        {
            "decision": "reject",
            "baseline_score": -0.6,
            "candidate_score": -0.9,
            "baseline_drawdown": 0.1,
            "candidate_drawdown": 0.12,
            "baseline_closed_trades": 3,
            "candidate_closed_trades": 2,
        },
    ]

    summary = summarize_reports(reports)

    assert summary["kept_windows"] == 1
    assert summary["rejected_windows"] == 1
    assert summary["average_baseline_score"] == pytest.approx(-0.7)
    assert summary["average_candidate_score"] == pytest.approx(-0.8)


def test_decide_robustness_keeps_strong_candidate() -> None:
    windows_results = [
        {
            "decision": "keep",
            "baseline_score": -0.8,
            "candidate_score": -0.6,
            "baseline_drawdown": 0.2,
            "candidate_drawdown": 0.18,
            "baseline_closed_trades": 4,
            "candidate_closed_trades": 4,
        },
        {
            "decision": "keep",
            "baseline_score": -0.7,
            "candidate_score": -0.5,
            "baseline_drawdown": 0.2,
            "candidate_drawdown": 0.19,
            "baseline_closed_trades": 4,
            "candidate_closed_trades": 3,
        },
        {
            "decision": "keep",
            "baseline_score": -0.9,
            "candidate_score": -0.7,
            "baseline_drawdown": 0.3,
            "candidate_drawdown": 0.25,
            "baseline_closed_trades": 4,
            "candidate_closed_trades": 3,
        },
        {
            "decision": "reject",
            "baseline_score": -0.4,
            "candidate_score": -0.5,
            "baseline_drawdown": 0.1,
            "candidate_drawdown": 0.1,
            "baseline_closed_trades": 3,
            "candidate_closed_trades": 2,
        },
    ]

    decision, reason = decide_robustness(windows_results=windows_results)

    assert decision == "keep"
    assert "enough windows" in reason


def test_decide_robustness_rejects_too_few_kept_windows() -> None:
    windows_results = [
        {
            "decision": "keep",
            "baseline_score": -0.8,
            "candidate_score": -0.6,
            "baseline_drawdown": 0.2,
            "candidate_drawdown": 0.18,
            "baseline_closed_trades": 4,
            "candidate_closed_trades": 4,
        },
        {
            "decision": "reject",
            "baseline_score": -0.7,
            "candidate_score": -0.9,
            "baseline_drawdown": 0.2,
            "candidate_drawdown": 0.22,
            "baseline_closed_trades": 4,
            "candidate_closed_trades": 4,
        },
        {
            "decision": "reject",
            "baseline_score": -0.5,
            "candidate_score": -0.8,
            "baseline_drawdown": 0.1,
            "candidate_drawdown": 0.12,
            "baseline_closed_trades": 4,
            "candidate_closed_trades": 4,
        },
        {
            "decision": "reject",
            "baseline_score": -0.4,
            "candidate_score": -0.7,
            "baseline_drawdown": 0.1,
            "candidate_drawdown": 0.11,
            "baseline_closed_trades": 4,
            "candidate_closed_trades": 4,
        },
    ]

    decision, reason = decide_robustness(windows_results=windows_results)

    assert decision == "reject"
    assert "enough windows" in reason
