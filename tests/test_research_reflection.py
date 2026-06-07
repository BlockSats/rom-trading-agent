from trading_agent.research_reflection import (
    build_research_reflection_report,
    decide_reflection,
)


def test_decide_reflection_keeps_improved_score_without_higher_drawdown() -> None:
    baseline = {"score": -0.8, "max_drawdown": 0.20}
    candidate = {"score": -0.4, "max_drawdown": 0.18}

    decision, reason = decide_reflection(baseline, candidate)

    assert decision == "keep"
    assert "improved" in reason.lower()


def test_decide_reflection_rejects_higher_drawdown() -> None:
    baseline = {"score": -0.8, "max_drawdown": 0.20}
    candidate = {"score": -0.4, "max_drawdown": 0.25}

    decision, reason = decide_reflection(baseline, candidate)

    assert decision == "reject"
    assert "safely" in reason.lower()


def test_decide_reflection_rejects_weaker_score() -> None:
    baseline = {"score": -0.4, "max_drawdown": 0.20}
    candidate = {"score": -0.8, "max_drawdown": 0.18}

    decision, reason = decide_reflection(baseline, candidate)

    assert decision == "reject"
    assert "safely" in reason.lower()


def test_build_research_reflection_report() -> None:
    baseline = {
        "score": -0.8,
        "total_return": -0.10,
        "total_net_pnl": -100.0,
        "closed_trades": 10,
        "max_drawdown": 0.20,
        "sharpe_like": -2.0,
    }
    candidate = {
        "score": -0.4,
        "total_return": -0.05,
        "total_net_pnl": -50.0,
        "closed_trades": 8,
        "max_drawdown": 0.18,
        "sharpe_like": -1.0,
    }
    proposal = {
        "variable": "entry.threshold",
        "old_value": 30.0,
        "new_value": 28.0,
        "reason": "Fallback reflection adjusted one parameter after weak score.",
        "expected_effect": "Reduce weak entries.",
        "confidence": 0.50,
    }

    report = build_research_reflection_report(
        symbol="BTCUSDT",
        interval="1h",
        limit=500,
        baseline_report=baseline,
        proposal=proposal,
        candidate_report=candidate,
    )

    assert report["command"] == "research-cycle"
    assert report["reflection_enabled"] is True
    assert report["status"] == "completed"
    assert report["decision"] == "keep"
    assert report["hypothesis"]["variable"] == "entry.threshold"
    assert report["hypothesis"]["old_value"] == 30.0
    assert report["hypothesis"]["new_value"] == 28.0
    assert report["baseline"]["score"] == -0.8
    assert report["candidate"]["score"] == -0.4
