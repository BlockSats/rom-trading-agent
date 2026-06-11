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


def test_show_research_report_displays_full_report(tmp_path: Path) -> None:
    report_path = tmp_path / "research_cycle_report.json"
    report_path.write_text(
        json.dumps(
            {
                "command": "research-cycle",
                "status": "completed",
                "symbol": "BTCUSDT",
                "interval": "1h",
                "limit": 500,
                "csv_path": "data/BTCUSDT_1h.csv",
                "rows": 500,
                "gaps_detected": 0,
                "asset": "BTC/USDT",
                "timeframe": "1h",
                "initial_balance": 10000.0,
                "total_trades": 12,
                "final_balance": 10250.0,
                "net_pnl": 250.0,
                "winrate": 0.58,
                "profit_factor": 1.4,
                "expectancy": 20.8,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["show-research-report", "--path", str(report_path)])

    assert result.exit_code == 0
    assert "Research cycle report" in result.stdout
    assert "Command: research-cycle" in result.stdout
    assert "Status: completed" in result.stdout
    assert "Symbol: BTCUSDT" in result.stdout
    assert "Interval: 1h" in result.stdout
    assert "Limit: 500" in result.stdout
    assert "CSV path: data/BTCUSDT_1h.csv" in result.stdout
    assert "Rows: 500" in result.stdout
    assert "Gaps detected: 0" in result.stdout
    assert "Asset: BTC/USDT" in result.stdout
    assert "Timeframe: 1h" in result.stdout
    assert "Initial balance: 10000.0" in result.stdout
    assert "Total trades: 12" in result.stdout
    assert "Final balance: 10250.0" in result.stdout
    assert "Net PnL: 250.0" in result.stdout
    assert "Winrate: 0.58" in result.stdout
    assert "Profit factor: 1.4" in result.stdout
    assert "Expectancy: 20.8" in result.stdout


def test_show_research_report_displays_score_and_classification(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"status": "completed", "score": 87.5, "classification": "candidate"}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["show-research-report", "--path", str(report_path)])

    assert result.exit_code == 0
    assert "Score: 87.5" in result.stdout
    assert "Classification: candidate" in result.stdout


def test_show_research_report_partial_fields(tmp_path: Path) -> None:
    report_path = tmp_path / "partial.json"
    report_path.write_text(
        json.dumps({"symbol": "ETHUSDT", "rows": 200}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["show-research-report", "--path", str(report_path)])

    assert result.exit_code == 0
    assert "Symbol: ETHUSDT" in result.stdout
    assert "Rows: 200" in result.stdout
    assert "Command:" not in result.stdout
    assert "Status:" not in result.stdout


def test_show_research_report_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    result = runner.invoke(app, ["show-research-report", "--path", str(missing)])

    assert result.exit_code == 1
    assert "Research report not found" in result.stderr


def test_show_research_report_invalid_json(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not-json", encoding="utf-8")

    result = runner.invoke(app, ["show-research-report", "--path", str(bad_path)])

    assert result.exit_code == 1
    assert "Invalid research report JSON" in result.stderr


def test_show_research_report_default_path_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["show-research-report"])

    assert result.exit_code == 1
    assert "Research report not found" in result.stderr


def test_show_backtest_report_displays_full_report(tmp_path: Path) -> None:
    report_path = tmp_path / "backtest_report.json"
    report_path.write_text(
        json.dumps(
            {
                "csv_path": "data/BTCUSDT_1h.csv",
                "asset": "BTC/USDT",
                "timeframe": "1h",
                "rows": 500,
                "gaps_detected": 0,
                "initial_balance": 10000.0,
                "total_trades": 12,
                "final_balance": 10250.0,
                "net_pnl": 250.0,
                "winrate": 0.58,
                "profit_factor": 1.4,
                "expectancy": 20.8,
                "score": 72.5,
                "classification": "candidate",
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["show-backtest-report", "--path", str(report_path)])

    assert result.exit_code == 0
    assert "Backtest report" in result.stdout
    assert "CSV path: data/BTCUSDT_1h.csv" in result.stdout
    assert "Asset: BTC/USDT" in result.stdout
    assert "Timeframe: 1h" in result.stdout
    assert "Rows: 500" in result.stdout
    assert "Gaps detected: 0" in result.stdout
    assert "Initial balance: 10000.0" in result.stdout
    assert "Total trades: 12" in result.stdout
    assert "Final balance: 10250.0" in result.stdout
    assert "Net PnL: 250.0" in result.stdout
    assert "Winrate: 0.58" in result.stdout
    assert "Profit factor: 1.4" in result.stdout
    assert "Expectancy: 20.8" in result.stdout
    assert "Score: 72.5" in result.stdout
    assert "Classification: candidate" in result.stdout


def test_show_backtest_report_partial_fields(tmp_path: Path) -> None:
    report_path = tmp_path / "partial.json"
    report_path.write_text(
        json.dumps({"asset": "ETH/USDT", "rows": 200, "score": 55.0}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["show-backtest-report", "--path", str(report_path)])

    assert result.exit_code == 0
    assert "Backtest report" in result.stdout
    assert "Asset: ETH/USDT" in result.stdout
    assert "Rows: 200" in result.stdout
    assert "Score: 55.0" in result.stdout
    assert "CSV path:" not in result.stdout
    assert "Total trades:" not in result.stdout
    assert "Classification:" not in result.stdout


def test_show_backtest_report_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    result = runner.invoke(app, ["show-backtest-report", "--path", str(missing)])

    assert result.exit_code == 1
    assert "Backtest report not found" in result.stderr


def test_show_backtest_report_invalid_json(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not-json", encoding="utf-8")

    result = runner.invoke(app, ["show-backtest-report", "--path", str(bad_path)])

    assert result.exit_code == 1
    assert "Invalid backtest report JSON" in result.stderr


def test_show_backtest_report_non_object_json(tmp_path: Path) -> None:
    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps([{"csv_path": "data.csv"}]), encoding="utf-8")

    result = runner.invoke(app, ["show-backtest-report", "--path", str(list_path)])

    assert result.exit_code == 1
    assert "Invalid backtest report JSON" in result.stderr


def test_show_backtest_report_default_path_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["show-backtest-report"])

    assert result.exit_code == 1
    assert "Backtest report not found" in result.stderr
