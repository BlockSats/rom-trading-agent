from __future__ import annotations

from typing import Any


def decide_reflection(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
) -> tuple[str, str]:
    """Decide whether a controlled reflection candidate is worth keeping.

    The rule is intentionally conservative:
    - the candidate score must improve;
    - the candidate drawdown must not increase.
    """

    baseline_score = float(baseline_report.get("score", 0.0))
    candidate_score = float(candidate_report.get("score", 0.0))

    baseline_drawdown = float(baseline_report.get("max_drawdown", 0.0))
    candidate_drawdown = float(candidate_report.get("max_drawdown", 0.0))

    if candidate_score > baseline_score and candidate_drawdown <= baseline_drawdown:
        return "keep", "Candidate score improved without increasing drawdown."

    return "reject", "Candidate did not improve score safely enough."


def build_research_reflection_report(
    *,
    symbol: str,
    interval: str,
    limit: int,
    baseline_report: dict[str, Any],
    proposal: dict[str, Any],
    candidate_report: dict[str, Any],
) -> dict[str, Any]:
    """Build a comparison report for baseline vs one reflected candidate."""

    decision, decision_reason = decide_reflection(
        baseline_report=baseline_report,
        candidate_report=candidate_report,
    )

    return {
        "command": "research-cycle",
        "reflection_enabled": True,
        "status": "completed",
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
        "baseline": {
            "score": baseline_report.get("score"),
            "total_return": baseline_report.get("total_return"),
            "total_net_pnl": baseline_report.get("total_net_pnl"),
            "closed_trades": baseline_report.get("closed_trades"),
            "max_drawdown": baseline_report.get("max_drawdown"),
            "sharpe_like": baseline_report.get("sharpe_like"),
        },
        "hypothesis": {
            "variable": proposal["variable"],
            "old_value": proposal["old_value"],
            "new_value": proposal["new_value"],
            "reason": proposal.get("reason"),
            "expected_effect": proposal.get("expected_effect"),
            "confidence": proposal.get("confidence"),
        },
        "candidate": {
            "score": candidate_report.get("score"),
            "total_return": candidate_report.get("total_return"),
            "total_net_pnl": candidate_report.get("total_net_pnl"),
            "closed_trades": candidate_report.get("closed_trades"),
            "max_drawdown": candidate_report.get("max_drawdown"),
            "sharpe_like": candidate_report.get("sharpe_like"),
        },
        "decision": decision,
        "decision_reason": decision_reason,
    }
