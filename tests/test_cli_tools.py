from __future__ import annotations

import json

from typer.testing import CliRunner

from trading_agent.cli import app


runner = CliRunner()


def test_status_command() -> None:
    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)

    assert payload["project"] == "trading-agent"
    assert "strategy" in payload
    assert "goal" in payload
    assert "state" in payload


def test_config_command_all() -> None:
    result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)

    assert "strategy" in payload
    assert "goal" in payload


def test_config_command_strategy_only() -> None:
    result = runner.invoke(app, ["config", "--section", "strategy"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)

    assert "strategy" in payload
    assert "goal" not in payload


def test_config_command_goal_only() -> None:
    result = runner.invoke(app, ["config", "--section", "goal"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)

    assert "goal" in payload
    assert "strategy" not in payload


def test_config_command_invalid_section() -> None:
    result = runner.invoke(app, ["config", "--section", "bad"])

    assert result.exit_code == 1
    assert "Invalid section" in result.stderr
