from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from trading_agent.cli import app


runner = CliRunner()


def _write_ema_atr_config(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "trades.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "state" / "hypotheses.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "config" / "strategy.yaml").write_text(
        """
version: "0001"
strategy_id: ema_atr_trend
asset: "BTC/USDT"
timeframe: "1h"
entry:
  indicator: "ema_atr"
  fast_ema_period: 2
  slow_ema_period: 4
  direction: "long"
exit:
  atr_period: 2
  atr_stop_multiplier: 1.5
risk:
  stop_loss_pct: 2.0
  position_size_pct: 10.0
costs:
  fee_pct: 0.10
  slippage_pct: 0.05
reflection:
  one_variable_only: true
  allowed_variables:
    - "entry.fast_ema_period"
""",
        encoding="utf-8",
    )
    (tmp_path / "config" / "goal.yaml").write_text(
        """
asset: "BTC/USDT"
target_return_30d: 0.05
max_drawdown: 0.08
min_sharpe: 1.2
reflection_every_closed_trades: 5
""",
        encoding="utf-8",
    )


def test_status_command() -> None:
    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)

    assert payload["project"] == "trading-agent"
    assert "strategy" in payload
    assert "goal" in payload
    assert "state" in payload


def test_config_command_default_readable() -> None:
    result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    assert "Strategy" in result.stdout
    assert "Strategy ID: rsi_baseline" in result.stdout
    assert "Version:" in result.stdout
    assert "Goal" in result.stdout
    assert "Target return 30d:" in result.stdout


def test_config_command_default_readable_for_ema_atr_trend(tmp_path: Path, monkeypatch) -> None:
    _write_ema_atr_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    assert "Strategy ID: ema_atr_trend" in result.stdout
    assert "Entry: EMA 2/4 trend" in result.stdout
    assert "Exit: ATR 2 x 1.5" in result.stdout


def test_config_command_all() -> None:
    result = runner.invoke(app, ["config", "--section", "all"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)

    assert "strategy" in payload
    assert "goal" in payload


def test_config_command_strategy_only() -> None:
    result = runner.invoke(app, ["config", "--section", "strategy"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)

    assert "strategy" in payload
    assert payload["strategy"]["strategy_id"] == "rsi_baseline"
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


def test_research_policy_command_readable() -> None:
    result = runner.invoke(app, ["research-policy"])

    assert result.exit_code == 0
    assert "Research policy" in result.stdout
    assert "Comparison acceptance" in result.stdout
    assert "Min total trades:" in result.stdout
    assert "Min windows with trades:" in result.stdout
    assert "Min positive expectancy windows:" in result.stdout
    assert "Min average expectancy:" in result.stdout
    assert "Min average profit factor:" in result.stdout


def test_research_policy_command_shows_configured_values(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "research_policy.yaml").write_text(
        """
comparison_acceptance:
  min_total_trades: 20
  min_windows_with_trades: 3
  min_positive_expectancy_windows: 3
  min_average_expectancy: 0.5
  min_average_profit_factor: 1.3
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["research-policy"])

    assert result.exit_code == 0
    assert "Min total trades: 20" in result.stdout
    assert "Min windows with trades: 3" in result.stdout
    assert "Min positive expectancy windows: 3" in result.stdout
    assert "Min average expectancy: 0.5" in result.stdout
    assert "Min average profit factor: 1.3" in result.stdout
