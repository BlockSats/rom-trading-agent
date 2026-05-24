from __future__ import annotations

from pathlib import Path

import pytest

from trading_agent.config import load_goal, load_strategy, validate_strategy


def test_valid_strategy_loads(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "strategy.yaml").write_text(
        """
version: "0001"
asset: "BTC/USDT"
timeframe: "1h"
entry:
  indicator: "rsi"
  threshold: 30
  direction: "long"
exit:
  rsi_take_profit: 55
risk:
  stop_loss_pct: 2.0
  position_size_pct: 10.0
costs:
  fee_pct: 0.10
  slippage_pct: 0.05
reflection:
  one_variable_only: true
  allowed_variables:
    - "entry.threshold"
""",
        encoding="utf-8",
    )
    loaded = load_strategy(config_dir / "strategy.yaml")
    assert loaded["version"] == "0001"
    assert loaded["entry"]["threshold"] == 30


def test_valid_goal_loads(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "goal.yaml").write_text(
        """
asset: "BTC/USDT"
target_return_30d: 0.05
max_drawdown: 0.08
min_sharpe: 1.2
reflection_every_closed_trades: 5
""",
        encoding="utf-8",
    )
    loaded = load_goal(config_dir / "goal.yaml")
    assert loaded["asset"] == "BTC/USDT"
    assert loaded["max_drawdown"] == 0.08


def test_invalid_missing_strategy_fields_raise_errors() -> None:
    with pytest.raises(ValueError):
        validate_strategy({"version": "0001"})


def test_invalid_risk_values_raise_errors() -> None:
    valid = {
        "version": "0001",
        "asset": "BTC/USDT",
        "timeframe": "1h",
        "entry": {"indicator": "rsi", "threshold": 30, "direction": "long"},
        "exit": {"rsi_take_profit": 55},
        "risk": {"stop_loss_pct": 0, "position_size_pct": 10},
        "costs": {"fee_pct": 0.1, "slippage_pct": 0.05},
        "reflection": {"one_variable_only": True, "allowed_variables": ["entry.threshold"]},
    }
    with pytest.raises(ValueError):
        validate_strategy(valid)


def test_invalid_costs_raise_errors() -> None:
    valid = {
        "version": "0001",
        "asset": "BTC/USDT",
        "timeframe": "1h",
        "entry": {"indicator": "rsi", "threshold": 30, "direction": "long"},
        "exit": {"rsi_take_profit": 55},
        "risk": {"stop_loss_pct": 2, "position_size_pct": 10},
        "costs": {"fee_pct": -0.1, "slippage_pct": 0.05},
        "reflection": {"one_variable_only": True, "allowed_variables": ["entry.threshold"]},
    }
    with pytest.raises(ValueError):
        validate_strategy(valid)


def test_reflection_one_variable_only_must_be_true() -> None:
    valid = {
        "version": "0001",
        "asset": "BTC/USDT",
        "timeframe": "1h",
        "entry": {"indicator": "rsi", "threshold": 30, "direction": "long"},
        "exit": {"rsi_take_profit": 55},
        "risk": {"stop_loss_pct": 2, "position_size_pct": 10},
        "costs": {"fee_pct": 0.1, "slippage_pct": 0.05},
        "reflection": {"one_variable_only": False, "allowed_variables": ["entry.threshold"]},
    }
    with pytest.raises(ValueError):
        validate_strategy(valid)
