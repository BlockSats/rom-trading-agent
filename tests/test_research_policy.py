from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trading_agent.cli import classify_strategy_summary
from trading_agent.config import load_research_policy, validate_research_policy


def test_valid_research_policy_loads() -> None:
    policy = load_research_policy()

    thresholds = policy["comparison_acceptance"]
    assert thresholds["min_total_trades"] == 10
    assert thresholds["min_windows_with_trades"] == 2
    assert thresholds["min_positive_expectancy_windows"] == 2
    assert thresholds["min_average_expectancy"] == 0.0
    assert thresholds["min_average_profit_factor"] == 1.1


def test_research_policy_rejects_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "research_policy.yaml"
    path.write_text("comparison_acceptance: [", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_research_policy(path)


def test_research_policy_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "research_policy.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping"):
        load_research_policy(path)


def test_research_policy_rejects_non_numeric_threshold() -> None:
    with pytest.raises(ValueError, match="min_total_trades must be numeric"):
        validate_research_policy(
            {
                "comparison_acceptance": {
                    "min_total_trades": "ten",
                    "min_windows_with_trades": 2,
                    "min_positive_expectancy_windows": 2,
                    "min_average_expectancy": 0.0,
                    "min_average_profit_factor": 1.1,
                }
            }
        )


def test_default_research_policy_preserves_v023_classification() -> None:
    policy = load_research_policy()

    assert classify_strategy_summary(
        {
            "total_trades": 10,
            "windows_with_trades": 2,
            "positive_expectancy_windows": 2,
            "average_expectancy": 0.1,
            "average_profit_factor": 1.1,
        },
        policy,
    ) == "candidate"
