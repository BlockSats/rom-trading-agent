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


# --- show-assets-comparison-report ---

_ASSETS_REPORT = {
    "command": "compare-strategies-assets-csv",
    "windows": 4,
    "assets": ["BTCUSDT", "ETHUSDT"],
    "strategies": ["rsi_baseline", "ema_atr_trend"],
    "output_dir": "outputs",
    "report_path": "outputs/strategy_assets_comparison_report.json",
    "results_by_asset": {
        "BTCUSDT": {
            "csv_path": "data/BTCUSDT_1h.csv",
            "rows": 80,
            "gaps_detected": 0,
            "summary_by_strategy": [
                {
                    "strategy_id": "rsi_baseline",
                    "windows": 4,
                    "windows_with_trades": 3,
                    "total_trades": 12,
                    "positive_expectancy_windows": 2,
                    "average_expectancy": 75.0,
                    "average_profit_factor": 1.4,
                },
                {
                    "strategy_id": "ema_atr_trend",
                    "windows": 4,
                    "windows_with_trades": 2,
                    "total_trades": 8,
                    "positive_expectancy_windows": 1,
                    "average_expectancy": 30.0,
                    "average_profit_factor": 1.2,
                },
            ],
        },
        "ETHUSDT": {
            "csv_path": "data/ETHUSDT_1h.csv",
            "rows": 60,
            "gaps_detected": 1,
            "summary_by_strategy": [
                {
                    "strategy_id": "rsi_baseline",
                    "windows": 4,
                    "windows_with_trades": 4,
                    "total_trades": 16,
                    "positive_expectancy_windows": 3,
                    "average_expectancy": 90.0,
                    "average_profit_factor": 1.6,
                },
            ],
        },
    },
}


def test_show_assets_comparison_report_full(tmp_path: Path) -> None:
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    assert result.exit_code == 0
    assert "Assets comparison report" in result.stdout
    assert "Command: compare-strategies-assets-csv" in result.stdout
    assert "Windows: 4" in result.stdout
    assert "Output dir: outputs" in result.stdout
    assert "Report path: outputs/strategy_assets_comparison_report.json" in result.stdout
    assert "Results by asset" in result.stdout
    assert "BTCUSDT" in result.stdout
    assert "CSV path: data/BTCUSDT_1h.csv" in result.stdout
    assert "Rows: 80" in result.stdout
    assert "Gaps detected: 0" in result.stdout
    assert "Summary by strategy" in result.stdout
    assert "rsi_baseline" in result.stdout
    assert "Total trades: 12" in result.stdout
    assert "ETHUSDT" in result.stdout
    assert "Rows: 60" in result.stdout
    assert "Gaps detected: 1" in result.stdout


def test_show_assets_comparison_report_custom_path(tmp_path: Path) -> None:
    report_path = tmp_path / "custom_assets.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    assert result.exit_code == 0
    assert "Assets comparison report" in result.stdout


def test_show_assets_comparison_report_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(missing)])

    assert result.exit_code == 1
    assert "Assets comparison report not found" in result.stderr


def test_show_assets_comparison_report_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not valid json {{{", encoding="utf-8")

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(bad)])

    assert result.exit_code == 1
    assert "Invalid assets comparison report JSON" in result.stderr


def test_show_assets_comparison_report_json_not_dict(tmp_path: Path) -> None:
    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps([{"csv_path": "data.csv"}]), encoding="utf-8")

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(list_path)])

    assert result.exit_code == 1
    assert "Invalid assets comparison report JSON" in result.stderr


def test_show_assets_comparison_report_missing_results_by_asset(tmp_path: Path) -> None:
    report_path = tmp_path / "no_results.json"
    report_path.write_text(json.dumps({"command": "compare-strategies-assets-csv"}), encoding="utf-8")

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    assert result.exit_code == 1
    assert "missing results_by_asset" in result.stderr


def test_show_assets_comparison_report_results_by_asset_not_dict(tmp_path: Path) -> None:
    report_path = tmp_path / "bad_results.json"
    report_path.write_text(
        json.dumps({"command": "compare-strategies-assets-csv", "results_by_asset": ["BTCUSDT"]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    assert result.exit_code == 1
    assert "missing results_by_asset" in result.stderr


def test_show_assets_comparison_report_asset_order(tmp_path: Path) -> None:
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    assert result.exit_code == 0
    btc_pos = result.stdout.index("BTCUSDT")
    eth_pos = result.stdout.index("ETHUSDT")
    assert btc_pos < eth_pos


def test_show_assets_comparison_report_strategy_order(tmp_path: Path) -> None:
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    assert result.exit_code == 0
    rsi_pos = result.stdout.index("rsi_baseline")
    ema_pos = result.stdout.index("ema_atr_trend")
    assert rsi_pos < ema_pos


def test_show_assets_comparison_report_no_forbidden_keys(tmp_path: Path) -> None:
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    assert result.exit_code == 0
    for forbidden in ["best_strategy", "best_asset", "winner", "rank", "ranking", "global_score", "selected_strategy"]:
        assert forbidden not in result.stdout


_MINIMAL_RESEARCH_POLICY_YAML = """\
comparison_acceptance:
  min_total_trades: 10
  min_windows_with_trades: 2
  min_positive_expectancy_windows: 2
  min_average_expectancy: 0.0
  min_average_profit_factor: 1.1
"""


def _write_research_policy(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "research_policy.yaml").write_text(_MINIMAL_RESEARCH_POLICY_YAML, encoding="utf-8")


def test_show_assets_comparison_report_does_not_modify_trades(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_research_policy(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    trades_file = state_dir / "trades.jsonl"
    trades_file.write_text('{"trade": 1}\n', encoding="utf-8")
    original = trades_file.read_text(encoding="utf-8")

    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")
    runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    assert trades_file.read_text(encoding="utf-8") == original


def test_show_assets_comparison_report_does_not_modify_hypotheses(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_research_policy(tmp_path)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    hyp_file = state_dir / "hypotheses.jsonl"
    hyp_file.write_text('{"hypothesis": 1}\n', encoding="utf-8")
    original = hyp_file.read_text(encoding="utf-8")

    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")
    runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    assert hyp_file.read_text(encoding="utf-8") == original


def test_show_assets_comparison_report_no_summary_by_strategy(tmp_path: Path) -> None:
    report = {
        "command": "compare-strategies-assets-csv",
        "windows": 2,
        "results_by_asset": {
            "BTCUSDT": {
                "csv_path": "data/BTCUSDT_1h.csv",
                "rows": 40,
                "gaps_detected": 0,
            }
        },
    }
    report_path = tmp_path / "no_summary.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    assert result.exit_code == 0
    assert "BTCUSDT" in result.stdout
    assert "Summary by strategy" not in result.stdout


def test_show_assets_comparison_report_default_path_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["show-assets-comparison-report"])

    assert result.exit_code == 1
    assert "Assets comparison report not found" in result.stderr


# --- v0.36: research status, reasons, window robustness diagnostics ---

_ASSETS_REPORT_WITH_RESULTS = {
    "command": "compare-strategies-assets-csv",
    "windows": 2,
    "assets": ["BTCUSDT"],
    "strategies": ["rsi_baseline"],
    "output_dir": "outputs",
    "report_path": "outputs/strategy_assets_comparison_report.json",
    "results_by_asset": {
        "BTCUSDT": {
            "csv_path": "data/BTCUSDT_1h.csv",
            "rows": 40,
            "gaps_detected": 0,
            "results": [
                {
                    "window": 1,
                    "strategies": [
                        {
                            "strategy_id": "rsi_baseline",
                            "closed_trades": 6,
                            "expectancy": 50.0,
                            "profit_factor": 1.3,
                        }
                    ],
                },
                {
                    "window": 2,
                    "strategies": [
                        {
                            "strategy_id": "rsi_baseline",
                            "closed_trades": 6,
                            "expectancy": -10.0,
                            "profit_factor": 0.9,
                        }
                    ],
                },
            ],
            "summary_by_strategy": [
                {
                    "strategy_id": "rsi_baseline",
                    "windows": 2,
                    "windows_with_trades": 2,
                    "total_trades": 12,
                    "positive_expectancy_windows": 1,
                    "average_expectancy": 20.0,
                    "average_profit_factor": 1.1,
                }
            ],
        }
    },
}


def test_show_assets_comparison_report_shows_research_status(tmp_path: Path) -> None:
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    assert result.exit_code == 0
    assert "Research status:" in result.stdout


def test_show_assets_comparison_report_shows_reasons(tmp_path: Path) -> None:
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    assert result.exit_code == 0
    assert "Reasons:" in result.stdout


def test_show_assets_comparison_report_shows_window_robustness_when_results_present(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT_WITH_RESULTS), encoding="utf-8")

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    assert result.exit_code == 0
    assert "Window robustness diagnostics" in result.stdout
    assert "Zero trade windows:" in result.stdout
    assert "Negative expectancy windows:" in result.stdout


def test_show_assets_comparison_report_no_window_robustness_when_results_absent(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    assert result.exit_code == 0
    assert "Window robustness diagnostics" not in result.stdout


def test_show_assets_comparison_report_no_results_does_not_crash(tmp_path: Path) -> None:
    report = {
        "command": "compare-strategies-assets-csv",
        "windows": 2,
        "results_by_asset": {
            "BTCUSDT": {
                "csv_path": "data/BTCUSDT_1h.csv",
                "rows": 40,
                "gaps_detected": 0,
                "summary_by_strategy": [
                    {
                        "strategy_id": "rsi_baseline",
                        "windows": 2,
                        "windows_with_trades": 2,
                        "total_trades": 12,
                        "positive_expectancy_windows": 2,
                        "average_expectancy": 50.0,
                        "average_profit_factor": 1.2,
                    }
                ],
            }
        },
    }
    report_path = tmp_path / "no_results.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    assert result.exit_code == 0
    assert "Research status:" in result.stdout
    assert "Window robustness diagnostics" not in result.stdout


def test_show_assets_comparison_report_no_backtest_launched(tmp_path: Path) -> None:
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    assert result.exit_code == 0
    assert "backtest" not in result.stdout.lower()


def test_show_assets_comparison_report_no_csv_load(tmp_path: Path) -> None:
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    assert result.exit_code == 0
    assert "Loading CSV" not in result.stdout
    assert "load_ohlcv" not in result.stdout


def test_show_assets_comparison_report_no_file_created(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")
    files_before = set(tmp_path.rglob("*"))

    runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    files_after = set(tmp_path.rglob("*"))
    assert files_after == files_before


def test_show_assets_comparison_report_no_forbidden_keys_extended(tmp_path: Path) -> None:
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT_WITH_RESULTS), encoding="utf-8")

    result = runner.invoke(app, ["show-assets-comparison-report", "--path", str(report_path)])

    assert result.exit_code == 0
    for forbidden in [
        "best_strategy",
        "best_asset",
        "winner",
        "rank",
        "ranking",
        "global_score",
        "selected_strategy",
    ]:
        assert forbidden not in result.stdout


# --- show-assets-comparison-report display filters (v0.37) ---


def test_show_assets_report_filter_single_asset(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        ["show-assets-comparison-report", "--path", str(report_path), "--asset", "BTCUSDT"],
    )

    assert result.exit_code == 0
    # "  BTCUSDT" (2 spaces) is the indented result entry
    assert "  BTCUSDT" in result.stdout
    assert "  ETHUSDT" not in result.stdout


def test_show_assets_report_filter_multiple_assets(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "show-assets-comparison-report",
            "--path", str(report_path),
            "--asset", "BTCUSDT",
            "--asset", "ETHUSDT",
        ],
    )

    assert result.exit_code == 0
    assert "BTCUSDT" in result.stdout
    assert "ETHUSDT" in result.stdout


def test_show_assets_report_filter_asset_order_preserved(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "show-assets-comparison-report",
            "--path", str(report_path),
            "--asset", "ETHUSDT",
            "--asset", "BTCUSDT",
        ],
    )

    assert result.exit_code == 0
    btc_pos = result.stdout.index("BTCUSDT")
    eth_pos = result.stdout.index("ETHUSDT")
    assert btc_pos < eth_pos


def test_show_assets_report_filter_single_strategy(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "show-assets-comparison-report",
            "--path", str(report_path),
            "--strategy", "rsi_baseline",
        ],
    )

    assert result.exit_code == 0
    # "      rsi_baseline" (6 spaces) is the indented strategy entry in results
    assert "      rsi_baseline" in result.stdout
    assert "      ema_atr_trend" not in result.stdout


def test_show_assets_report_filter_multiple_strategies(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "show-assets-comparison-report",
            "--path", str(report_path),
            "--strategy", "rsi_baseline",
            "--strategy", "ema_atr_trend",
        ],
    )

    assert result.exit_code == 0
    assert "rsi_baseline" in result.stdout
    assert "ema_atr_trend" in result.stdout


def test_show_assets_report_filter_strategy_order_preserved(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "show-assets-comparison-report",
            "--path", str(report_path),
            "--strategy", "ema_atr_trend",
            "--strategy", "rsi_baseline",
        ],
    )

    assert result.exit_code == 0
    rsi_pos = result.stdout.index("rsi_baseline")
    ema_pos = result.stdout.index("ema_atr_trend")
    assert rsi_pos < ema_pos


def test_show_assets_report_filter_combined_asset_strategy(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "show-assets-comparison-report",
            "--path", str(report_path),
            "--asset", "BTCUSDT",
            "--strategy", "rsi_baseline",
        ],
    )

    assert result.exit_code == 0
    assert "  BTCUSDT" in result.stdout
    assert "      rsi_baseline" in result.stdout
    assert "  ETHUSDT" not in result.stdout
    assert "      ema_atr_trend" not in result.stdout


def test_show_assets_report_filter_unknown_asset_exits(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "show-assets-comparison-report",
            "--path", str(report_path),
            "--asset", "XYZUSDT",
        ],
    )

    assert result.exit_code == 1
    assert "XYZUSDT" in result.stdout
    assert "not found" in result.stdout


def test_show_assets_report_filter_unknown_strategy_exits(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "show-assets-comparison-report",
            "--path", str(report_path),
            "--strategy", "unknown_strategy",
        ],
    )

    assert result.exit_code == 1
    assert "unknown_strategy" in result.stdout
    assert "not found" in result.stdout


def test_show_assets_report_filter_no_summary_strategy_filter_exits(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_without_summary = {
        "command": "compare-strategies-assets-csv",
        "windows": 4,
        "assets": ["BTCUSDT"],
        "strategies": ["rsi_baseline"],
        "output_dir": "outputs",
        "report_path": "outputs/strategy_assets_comparison_report.json",
        "results_by_asset": {
            "BTCUSDT": {
                "csv_path": "data/BTCUSDT_1h.csv",
                "rows": 80,
                "gaps_detected": 0,
            },
        },
    }
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(report_without_summary), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "show-assets-comparison-report",
            "--path", str(report_path),
            "--strategy", "rsi_baseline",
        ],
    )

    assert result.exit_code == 1
    assert "not found" in result.stdout


def test_show_assets_report_filter_no_forbidden_keys_with_filters(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "show-assets-comparison-report",
            "--path", str(report_path),
            "--asset", "BTCUSDT",
            "--strategy", "rsi_baseline",
        ],
    )

    assert result.exit_code == 0
    for forbidden in [
        "best_strategy",
        "best_asset",
        "winner",
        "rank",
        "ranking",
        "global_score",
        "selected_strategy",
    ]:
        assert forbidden not in result.stdout


def test_show_assets_report_filter_no_file_created(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")
    files_before = set(tmp_path.rglob("*"))

    runner.invoke(
        app,
        [
            "show-assets-comparison-report",
            "--path", str(report_path),
            "--asset", "BTCUSDT",
        ],
    )

    files_after = set(tmp_path.rglob("*"))
    assert files_after == files_before


def test_show_assets_report_filter_no_trades_modified(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_ASSETS_REPORT), encoding="utf-8")
    trades_path = tmp_path / "trades.jsonl"
    trades_path.write_text("", encoding="utf-8")
    mtime_before = trades_path.stat().st_mtime

    runner.invoke(
        app,
        [
            "show-assets-comparison-report",
            "--path", str(report_path),
            "--asset", "BTCUSDT",
        ],
    )

    assert trades_path.stat().st_mtime == mtime_before


# --- show-assets-comparison-report --list-assets / --list-strategies ---

_MULTI_STRATEGY_ASSETS_REPORT = {
    "command": "compare-strategies-assets-csv",
    "windows": 4,
    "assets": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "strategies": ["rsi_baseline", "ema_atr_trend", "donchian_breakout"],
    "output_dir": "outputs",
    "report_path": "outputs/strategy_assets_comparison_report.json",
    "results_by_asset": {
        "BTCUSDT": {
            "csv_path": "data/BTCUSDT_1h.csv",
            "rows": 80,
            "gaps_detected": 0,
            "summary_by_strategy": [
                {
                    "strategy_id": "rsi_baseline",
                    "windows": 4,
                    "windows_with_trades": 3,
                    "total_trades": 12,
                    "positive_expectancy_windows": 2,
                    "average_expectancy": 75.0,
                    "average_profit_factor": 1.4,
                },
                {
                    "strategy_id": "ema_atr_trend",
                    "windows": 4,
                    "windows_with_trades": 2,
                    "total_trades": 8,
                    "positive_expectancy_windows": 1,
                    "average_expectancy": 30.0,
                    "average_profit_factor": 1.2,
                },
            ],
        },
        "ETHUSDT": {
            "csv_path": "data/ETHUSDT_1h.csv",
            "rows": 60,
            "gaps_detected": 0,
            "summary_by_strategy": [
                {
                    "strategy_id": "ema_atr_trend",
                    "windows": 4,
                    "windows_with_trades": 2,
                    "total_trades": 6,
                    "positive_expectancy_windows": 1,
                    "average_expectancy": 20.0,
                    "average_profit_factor": 1.1,
                },
                {
                    "strategy_id": "donchian_breakout",
                    "windows": 4,
                    "windows_with_trades": 3,
                    "total_trades": 10,
                    "positive_expectancy_windows": 2,
                    "average_expectancy": 55.0,
                    "average_profit_factor": 1.3,
                },
            ],
        },
        "SOLUSDT": {
            "csv_path": "data/SOLUSDT_1h.csv",
            "rows": 50,
            "gaps_detected": 0,
            "summary_by_strategy": [
                {
                    "strategy_id": "donchian_breakout",
                    "windows": 4,
                    "windows_with_trades": 4,
                    "total_trades": 14,
                    "positive_expectancy_windows": 3,
                    "average_expectancy": 80.0,
                    "average_profit_factor": 1.5,
                },
            ],
        },
    },
}


def test_show_assets_report_list_assets_basic(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_MULTI_STRATEGY_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        ["show-assets-comparison-report", "--path", str(report_path), "--list-assets"],
    )

    assert result.exit_code == 0
    assert "Assets in report:" in result.stdout
    assert "BTCUSDT" in result.stdout
    assert "ETHUSDT" in result.stdout
    assert "SOLUSDT" in result.stdout


def test_show_assets_report_list_assets_order(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_MULTI_STRATEGY_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        ["show-assets-comparison-report", "--path", str(report_path), "--list-assets"],
    )

    assert result.exit_code == 0
    btc_pos = result.stdout.index("BTCUSDT")
    eth_pos = result.stdout.index("ETHUSDT")
    sol_pos = result.stdout.index("SOLUSDT")
    assert btc_pos < eth_pos < sol_pos


def test_show_assets_report_list_assets_no_full_report(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_MULTI_STRATEGY_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        ["show-assets-comparison-report", "--path", str(report_path), "--list-assets"],
    )

    assert result.exit_code == 0
    assert "Assets comparison report" not in result.stdout
    assert "Results by asset" not in result.stdout
    assert "Summary by strategy" not in result.stdout


def test_show_assets_report_list_strategies_basic(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_MULTI_STRATEGY_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        ["show-assets-comparison-report", "--path", str(report_path), "--list-strategies"],
    )

    assert result.exit_code == 0
    assert "Strategies in report:" in result.stdout
    assert "rsi_baseline" in result.stdout
    assert "ema_atr_trend" in result.stdout
    assert "donchian_breakout" in result.stdout


def test_show_assets_report_list_strategies_order(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_MULTI_STRATEGY_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        ["show-assets-comparison-report", "--path", str(report_path), "--list-strategies"],
    )

    assert result.exit_code == 0
    # rsi_baseline apparaît en premier (BTCUSDT), ema_atr_trend en second (BTCUSDT),
    # donchian_breakout en troisième (ETHUSDT) — ordre de première apparition
    rsi_pos = result.stdout.index("rsi_baseline")
    ema_pos = result.stdout.index("ema_atr_trend")
    don_pos = result.stdout.index("donchian_breakout")
    assert rsi_pos < ema_pos < don_pos


def test_show_assets_report_list_strategies_no_full_report(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_MULTI_STRATEGY_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        ["show-assets-comparison-report", "--path", str(report_path), "--list-strategies"],
    )

    assert result.exit_code == 0
    assert "Assets comparison report" not in result.stdout
    assert "Results by asset" not in result.stdout
    assert "Summary by strategy" not in result.stdout


def test_show_assets_report_list_assets_and_strategies(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_MULTI_STRATEGY_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "show-assets-comparison-report",
            "--path", str(report_path),
            "--list-assets",
            "--list-strategies",
        ],
    )

    assert result.exit_code == 0
    assert "Assets in report:" in result.stdout
    assert "Strategies in report:" in result.stdout
    assert "BTCUSDT" in result.stdout
    assert "rsi_baseline" in result.stdout
    assert "Assets comparison report" not in result.stdout


def test_show_assets_report_list_strategies_with_asset_filter(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_MULTI_STRATEGY_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "show-assets-comparison-report",
            "--path", str(report_path),
            "--asset", "ETHUSDT",
            "--list-strategies",
        ],
    )

    assert result.exit_code == 0
    assert "Strategies in report:" in result.stdout
    assert "ema_atr_trend" in result.stdout
    assert "donchian_breakout" in result.stdout
    # rsi_baseline n'est pas dans ETHUSDT
    assert "rsi_baseline" not in result.stdout


def test_show_assets_report_list_assets_with_strategy_filter(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_MULTI_STRATEGY_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "show-assets-comparison-report",
            "--path", str(report_path),
            "--strategy", "donchian_breakout",
            "--list-assets",
        ],
    )

    assert result.exit_code == 0
    assert "Assets in report:" in result.stdout
    assert "ETHUSDT" in result.stdout
    assert "SOLUSDT" in result.stdout
    # BTCUSDT ne contient pas donchian_breakout
    assert "BTCUSDT" not in result.stdout


def test_show_assets_report_list_unknown_asset_exits(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_MULTI_STRATEGY_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "show-assets-comparison-report",
            "--path", str(report_path),
            "--asset", "UNKNOWN",
            "--list-strategies",
        ],
    )

    assert result.exit_code == 1
    assert "Error" in result.stdout


def test_show_assets_report_list_unknown_strategy_exits(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_MULTI_STRATEGY_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "show-assets-comparison-report",
            "--path", str(report_path),
            "--strategy", "unknown_strategy",
            "--list-assets",
        ],
    )

    assert result.exit_code == 1
    assert "Error" in result.stdout


def test_show_assets_report_list_assets_no_summary(tmp_path: Path) -> None:
    report_no_summary = {
        "command": "compare-strategies-assets-csv",
        "windows": 2,
        "assets": ["BTCUSDT"],
        "strategies": [],
        "output_dir": "outputs",
        "report_path": "outputs/strategy_assets_comparison_report.json",
        "results_by_asset": {
            "BTCUSDT": {"csv_path": "data/BTCUSDT_1h.csv", "rows": 40, "gaps_detected": 0},
            "ETHUSDT": {"csv_path": "data/ETHUSDT_1h.csv", "rows": 30, "gaps_detected": 0},
        },
    }
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(report_no_summary), encoding="utf-8")

    result = runner.invoke(
        app,
        ["show-assets-comparison-report", "--path", str(report_path), "--list-assets"],
    )

    assert result.exit_code == 0
    assert "Assets in report:" in result.stdout
    assert "BTCUSDT" in result.stdout
    assert "ETHUSDT" in result.stdout


def test_show_assets_report_list_strategies_no_summary(tmp_path: Path) -> None:
    report_no_summary = {
        "command": "compare-strategies-assets-csv",
        "windows": 2,
        "assets": ["BTCUSDT"],
        "strategies": [],
        "output_dir": "outputs",
        "report_path": "outputs/strategy_assets_comparison_report.json",
        "results_by_asset": {
            "BTCUSDT": {"csv_path": "data/BTCUSDT_1h.csv", "rows": 40, "gaps_detected": 0},
        },
    }
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(report_no_summary), encoding="utf-8")

    result = runner.invoke(
        app,
        ["show-assets-comparison-report", "--path", str(report_path), "--list-strategies"],
    )

    assert result.exit_code == 0
    assert "No strategies found in report." in result.stdout


def test_show_assets_report_list_no_forbidden_keys(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_MULTI_STRATEGY_ASSETS_REPORT), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "show-assets-comparison-report",
            "--path", str(report_path),
            "--list-assets",
            "--list-strategies",
        ],
    )

    assert result.exit_code == 0
    forbidden = [
        "best_strategy", "best_asset", "winner", "rank", "ranking",
        "global_score", "selected_strategy",
    ]
    for key in forbidden:
        assert key not in result.stdout


def test_show_assets_report_list_no_file_created(tmp_path: Path) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_MULTI_STRATEGY_ASSETS_REPORT), encoding="utf-8")
    files_before = set(tmp_path.rglob("*"))

    runner.invoke(
        app,
        ["show-assets-comparison-report", "--path", str(report_path), "--list-assets"],
    )

    files_after = set(tmp_path.rglob("*"))
    assert files_after == files_before


def test_show_assets_report_list_no_trades_modified(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_research_policy(tmp_path)
    report_path = tmp_path / "assets_report.json"
    report_path.write_text(json.dumps(_MULTI_STRATEGY_ASSETS_REPORT), encoding="utf-8")
    trades_path = tmp_path / "trades.jsonl"
    trades_path.write_text("", encoding="utf-8")
    mtime_before = trades_path.stat().st_mtime

    runner.invoke(
        app,
        ["show-assets-comparison-report", "--path", str(report_path), "--list-assets"],
    )

    assert trades_path.stat().st_mtime == mtime_before
