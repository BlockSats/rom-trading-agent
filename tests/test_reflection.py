from __future__ import annotations

import pytest

from trading_agent.reflection import apply_reflection_proposal, propose_one_change


def _strategy() -> dict:
    return {
        "version": "0001",
        "strategy_id": "rsi_baseline",
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


def _ema_atr_strategy() -> dict:
    return {
        "version": "0001",
        "strategy_id": "ema_atr_trend",
        "asset": "BTC/USDT",
        "timeframe": "1h",
        "entry": {"indicator": "ema_atr", "fast_ema_period": 12, "slow_ema_period": 26, "direction": "long"},
        "exit": {"atr_period": 14, "atr_stop_multiplier": 2.0},
        "risk": {"stop_loss_pct": 2.0, "position_size_pct": 10.0},
        "costs": {"fee_pct": 0.1, "slippage_pct": 0.05},
        "reflection": {
            "one_variable_only": True,
            "allowed_variables": ["risk.stop_loss_pct", "risk.position_size_pct"],
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


def test_propose_one_change_rejects_unsupported_strategy_id() -> None:
    with pytest.raises(ValueError, match="reflection not supported for strategy: ema_atr_trend"):
        propose_one_change(_ema_atr_strategy(), {"score": -0.5})


def test_apply_reflection_proposal_rejects_unsupported_strategy_id() -> None:
    with pytest.raises(ValueError, match="reflection not supported for strategy: ema_atr_trend"):
        apply_reflection_proposal(_ema_atr_strategy(), {"variable": "risk.stop_loss_pct", "new_value": 1.5})


def _donchian_strategy() -> dict:
    return {
        "version": "0001",
        "strategy_id": "donchian_breakout",
        "asset": "BTC/USDT",
        "timeframe": "1h",
        "entry": {
            "indicator": "donchian",
            "donchian_period": 20,
            "direction": "long",
        },
        "exit": {"atr_period": 14, "atr_stop_multiplier": 2.0},
        "risk": {"stop_loss_pct": 2.0, "position_size_pct": 10.0},
        "costs": {"fee_pct": 0.1, "slippage_pct": 0.05},
        "reflection": {"one_variable_only": True, "allowed_variables": ["entry.donchian_period"]},
    }


def test_propose_one_change_rejects_donchian_breakout() -> None:
    with pytest.raises(ValueError, match="reflection not supported for strategy: donchian_breakout"):
        propose_one_change(_donchian_strategy(), {"score": -0.5})


def test_apply_reflection_proposal_rejects_donchian_breakout() -> None:
    with pytest.raises(ValueError, match="reflection not supported for strategy: donchian_breakout"):
        apply_reflection_proposal(_donchian_strategy(), {"variable": "entry.donchian_period", "new_value": 15})


def test_propose_one_change_without_strategy_id_defaults_to_rsi_baseline() -> None:
    strategy = _strategy()
    del strategy["strategy_id"]
    proposal = propose_one_change(strategy, {"score": -0.5})
    assert proposal["variable"] in strategy["reflection"]["allowed_variables"]

