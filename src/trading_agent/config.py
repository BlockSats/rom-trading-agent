from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_STRATEGY_IDS = {"rsi_baseline", "ema_atr_trend"}
DEFAULT_STRATEGY_ID = "rsi_baseline"


def load_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {file_path} must contain a mapping.")
    return data


def _is_numeric(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _require_mapping(data: Any, name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a mapping.")
    return data


def get_active_strategy_id(strategy: dict[str, Any]) -> str:
    strategy_id = strategy.get("strategy_id", DEFAULT_STRATEGY_ID)
    if not isinstance(strategy_id, str) or not strategy_id:
        raise ValueError("strategy.strategy_id must be a non-empty string")
    return strategy_id


def _require_positive_int(mapping: dict[str, Any], key: str, name: str) -> None:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name}.{key} must be a positive integer")


def _require_positive_number(mapping: dict[str, Any], key: str, name: str) -> None:
    value = mapping.get(key)
    if not _is_numeric(value) or float(value) <= 0:
        raise ValueError(f"{name}.{key} must be numeric and > 0")


def _validate_rsi_baseline(entry: dict[str, Any], exit_: dict[str, Any]) -> None:
    if "threshold" not in entry or not _is_numeric(entry["threshold"]):
        raise ValueError("strategy.entry.threshold must be numeric")
    if "rsi_take_profit" not in exit_ or not _is_numeric(exit_["rsi_take_profit"]):
        raise ValueError("strategy.exit.rsi_take_profit must be numeric")


def _validate_ema_atr_trend(entry: dict[str, Any], exit_: dict[str, Any]) -> None:
    _require_positive_int(entry, "fast_ema_period", "strategy.entry")
    _require_positive_int(entry, "slow_ema_period", "strategy.entry")
    if int(entry["fast_ema_period"]) >= int(entry["slow_ema_period"]):
        raise ValueError("strategy.entry.fast_ema_period must be lower than strategy.entry.slow_ema_period")
    _require_positive_int(exit_, "atr_period", "strategy.exit")
    _require_positive_number(exit_, "atr_stop_multiplier", "strategy.exit")


def validate_strategy(strategy: dict[str, Any]) -> dict[str, Any]:
    strategy = _require_mapping(strategy, "strategy")
    required_keys = ["version", "asset", "timeframe", "entry", "exit", "risk", "costs", "reflection"]
    for key in required_keys:
        if key not in strategy:
            raise ValueError(f"strategy missing required field: {key}")

    strategy_id = get_active_strategy_id(strategy)
    if strategy_id not in SUPPORTED_STRATEGY_IDS:
        raise ValueError(f"unsupported strategy_id: {strategy_id}")

    if not isinstance(strategy["version"], str) or not strategy["version"]:
        raise ValueError("strategy.version must be a non-empty string")
    if not isinstance(strategy["asset"], str) or not strategy["asset"]:
        raise ValueError("strategy.asset must be a non-empty string")
    if not isinstance(strategy["timeframe"], str) or not strategy["timeframe"]:
        raise ValueError("strategy.timeframe must be a non-empty string")

    entry = _require_mapping(strategy["entry"], "strategy.entry")
    exit_ = _require_mapping(strategy["exit"], "strategy.exit")
    risk = _require_mapping(strategy["risk"], "strategy.risk")
    costs = _require_mapping(strategy["costs"], "strategy.costs")
    reflection = _require_mapping(strategy["reflection"], "strategy.reflection")

    if strategy_id == "rsi_baseline":
        _validate_rsi_baseline(entry, exit_)
    elif strategy_id == "ema_atr_trend":
        _validate_ema_atr_trend(entry, exit_)
    if "stop_loss_pct" not in risk or not _is_numeric(risk["stop_loss_pct"]) or float(risk["stop_loss_pct"]) <= 0:
        raise ValueError("strategy.risk.stop_loss_pct must be numeric and > 0")
    if "position_size_pct" not in risk or not _is_numeric(risk["position_size_pct"]) or not (
        0 < float(risk["position_size_pct"]) <= 100
    ):
        raise ValueError("strategy.risk.position_size_pct must be numeric and between 0 and 100")
    if "fee_pct" not in costs or not _is_numeric(costs["fee_pct"]) or float(costs["fee_pct"]) < 0:
        raise ValueError("strategy.costs.fee_pct must be numeric and >= 0")
    if "slippage_pct" not in costs or not _is_numeric(costs["slippage_pct"]) or float(costs["slippage_pct"]) < 0:
        raise ValueError("strategy.costs.slippage_pct must be numeric and >= 0")
    if reflection.get("one_variable_only") is not True:
        raise ValueError("strategy.reflection.one_variable_only must be true")
    allowed_variables = reflection.get("allowed_variables")
    if not isinstance(allowed_variables, list) or not allowed_variables:
        raise ValueError("strategy.reflection.allowed_variables must be a non-empty list")

    return strategy


def validate_goal(goal: dict[str, Any]) -> dict[str, Any]:
    goal = _require_mapping(goal, "goal")
    required_keys = [
        "asset",
        "target_return_30d",
        "max_drawdown",
        "min_sharpe",
        "reflection_every_closed_trades",
    ]
    for key in required_keys:
        if key not in goal:
            raise ValueError(f"goal missing required field: {key}")

    if not isinstance(goal["asset"], str) or not goal["asset"]:
        raise ValueError("goal.asset must be a non-empty string")
    if not _is_numeric(goal["target_return_30d"]):
        raise ValueError("goal.target_return_30d must be numeric")
    if not _is_numeric(goal["max_drawdown"]) or not (0 <= float(goal["max_drawdown"]) <= 1):
        raise ValueError("goal.max_drawdown must be numeric and between 0 and 1")
    if not _is_numeric(goal["min_sharpe"]):
        raise ValueError("goal.min_sharpe must be numeric")
    if not isinstance(goal["reflection_every_closed_trades"], int) or goal["reflection_every_closed_trades"] <= 0:
        raise ValueError("goal.reflection_every_closed_trades must be a positive integer")

    return goal


def load_strategy(path: str | Path = "config/strategy.yaml") -> dict[str, Any]:
    return validate_strategy(load_yaml(path))


def load_goal(path: str | Path = "config/goal.yaml") -> dict[str, Any]:
    return validate_goal(load_yaml(path))
