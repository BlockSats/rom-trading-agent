from __future__ import annotations

import copy
from typing import Any


SAFE_BOUNDS = {
    "entry.threshold": (10, 45),
    "exit.rsi_take_profit": (45, 80),
    "risk.stop_loss_pct": (0.5, 10),
    "risk.position_size_pct": (1, 25),
}


def _get_nested(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        current = current[part]
    return current


def _set_nested(data: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = data
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def _adjust_value(variable: str, old_value: float, score_result: dict[str, Any]) -> float:
    low, high = SAFE_BOUNDS[variable]
    score = float(score_result.get("score", 0.0))

    if variable == "entry.threshold":
        if score <= 0:
            candidate = old_value - 2
        else:
            candidate = old_value + 1
    elif variable == "exit.rsi_take_profit":
        candidate = old_value + 1 if score <= 0 else old_value - 1
    elif variable == "risk.stop_loss_pct":
        candidate = old_value - 0.5 if score <= 0 else old_value + 0.5
    else:
        candidate = old_value - 1 if score <= 0 else old_value + 1

    if candidate <= low:
        candidate = min(high, low + (1 if isinstance(low, int) and isinstance(high, int) else 0.5))
    if candidate >= high:
        candidate = max(low, high - (1 if isinstance(low, int) and isinstance(high, int) else 0.5))
    return float(candidate)


def propose_one_change(strategy: dict[str, Any], score_result: dict[str, Any]) -> dict[str, Any]:
    allowed = list(strategy.get("reflection", {}).get("allowed_variables", []))
    if not allowed:
        raise ValueError("no allowed variables configured")

    variable = allowed[0]
    if variable not in SAFE_BOUNDS:
        raise ValueError(f"unsupported variable: {variable}")

    old_value = float(_get_nested(strategy, variable))
    new_value = _adjust_value(variable, old_value, score_result)
    if new_value == old_value:
        low, high = SAFE_BOUNDS[variable]
        new_value = low if old_value != low else high

    return {
        "variable": variable,
        "old_value": old_value,
        "new_value": new_value,
        "reason": "Fallback reflection adjusted one parameter after weak score.",
        "expected_effect": "Reduce weak entries.",
        "confidence": 0.50,
    }


def apply_reflection_proposal(strategy: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    strategy_copy = copy.deepcopy(strategy)
    reflection = strategy_copy.get("reflection", {})
    allowed = list(reflection.get("allowed_variables", []))

    variable = proposal.get("variable")
    if variable not in allowed:
        raise ValueError("proposed variable is not allowed")
    if variable not in SAFE_BOUNDS:
        raise ValueError("proposed variable is not supported")

    old_value = _get_nested(strategy_copy, variable)
    new_value = proposal.get("new_value")
    if old_value == new_value:
        raise ValueError("proposal must change exactly one variable")

    low, high = SAFE_BOUNDS[variable]
    if not isinstance(new_value, (int, float)) or not (low <= float(new_value) <= high):
        raise ValueError("proposed value is outside safe bounds")

    _set_nested(strategy_copy, variable, new_value)

    version = str(strategy_copy.get("version", "0000"))
    try:
        next_version = int(version) + 1
    except ValueError as exc:
        raise ValueError("strategy version must be numeric text") from exc
    strategy_copy["version"] = f"{next_version:04d}"
    return strategy_copy

