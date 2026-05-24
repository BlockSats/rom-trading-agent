from __future__ import annotations

import pytest

from trading_agent.reflection import apply_reflection_proposal, propose_one_change


def _strategy() -> dict:
    return {
        "version": "0001",
        "asset": "BTC/USDT",
        "timeframe": "1h",
        "entry": {"indicator": "rsi", "threshold": 30, "direction": "long"},
        "exit": {"rsi_take_profit": 55},
        "risk": {"stop_loss_pct": 2.0, "position_size_pct": 10.0},
        "costs": {"fee_pct": 0.1, "slippage_pct": 0.05},
        "reflection": {
            "one_variable_only": True,
            "allowed_variables": [
                "entry.threshold",
                "exit.rsi_take_profit",
                "risk.stop_loss_pct",
                "risk.position_size_pct",
            ],
        },
    }


def test_propose_one_change_returns_exactly_one_allowed_variable() -> None:
    proposal = propose_one_change(_strategy(), {"score": -0.5})
    assert proposal["variable"] in _strategy()["reflection"]["allowed_variables"]


def test_proposal_value_is_within_safe_bounds() -> None:
    proposal = propose_one_change(_strategy(), {"score": -0.5})
    assert 10 <= proposal["new_value"] <= 45


def test_apply_reflection_proposal_increments_version() -> None:
    updated = apply_reflection_proposal(_strategy(), {"variable": "entry.threshold", "new_value": 28})
    assert updated["version"] == "0002"
    assert updated["entry"]["threshold"] == 28


def test_disallowed_variable_is_rejected() -> None:
    with pytest.raises(ValueError):
        apply_reflection_proposal(_strategy(), {"variable": "not.allowed", "new_value": 28})


def test_out_of_bounds_value_is_rejected() -> None:
    with pytest.raises(ValueError):
        apply_reflection_proposal(_strategy(), {"variable": "entry.threshold", "new_value": 5})

