from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from trading_agent.cli import app, build_strategy_variant, classify_strategy_summary, get_classification_reasons
from trading_agent.config import load_research_policy, load_strategy
from trading_agent.data import generate_sample_ohlcv


def _write_research_policy(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "research_policy.yaml").write_text(
        """
comparison_acceptance:
  min_total_trades: 10
  min_windows_with_trades: 2
  min_positive_expectancy_windows: 2
  min_average_expectancy: 0.0
  min_average_profit_factor: 1.1
""",
        encoding="utf-8",
    )


def _write_config(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "trades.jsonl").write_text('{"sentinel": true}\n', encoding="utf-8")
    (tmp_path / "state" / "hypotheses.jsonl").write_text('{"sentinel": true}\n', encoding="utf-8")
    (tmp_path / "config" / "strategy.yaml").write_text(
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


def _write_csv(tmp_path: Path, with_gap: bool = True, rows: int = 50) -> Path:
    df = generate_sample_ohlcv(rows=rows, seed=42)
    if with_gap:
        df.loc[20:, "timestamp"] = df.loc[20:, "timestamp"] + pd.Timedelta(hours=1)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_backtest_cli_creates_outputs_and_leaves_state_untouched(tmp_path: Path, monkeypatch) -> None:
    _write_config(tmp_path)
    csv_path = _write_csv(tmp_path)
    monkeypatch.chdir(tmp_path)

    before_state = (tmp_path / "state" / "trades.jsonl").read_text(encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["backtest-csv", str(csv_path)])

    assert result.exit_code == 0, result.output
    report_path = tmp_path / "outputs" / "backtest_report.json"
    trades_path = tmp_path / "outputs" / "backtest_trades.jsonl"
    assert report_path.exists()
    assert trades_path.exists()
    assert (tmp_path / "state" / "trades.jsonl").read_text(encoding="utf-8") == before_state

    payload = json.loads(result.stdout)
    assert payload["output_dir"] == "outputs"
    assert payload["report_path"] == "outputs/backtest_report.json"
    assert payload["trades_path"] == "outputs/backtest_trades.jsonl"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["rows"] == 50
    assert "score" in report


def test_backtest_cli_writes_to_custom_output_dir(tmp_path: Path, monkeypatch) -> None:
    _write_config(tmp_path)
    csv_path = _write_csv(tmp_path, with_gap=False)
    monkeypatch.chdir(tmp_path)

    before_state = (tmp_path / "state" / "trades.jsonl").read_text(encoding="utf-8")
    custom_dir = tmp_path / "tmp" / "backtest-test"
    runner = CliRunner()
    result = runner.invoke(app, ["backtest-csv", str(csv_path), "--output-dir", str(custom_dir)])

    assert result.exit_code == 0, result.output
    report_path = custom_dir / "backtest_report.json"
    trades_path = custom_dir / "backtest_trades.jsonl"
    assert report_path.exists()
    assert trades_path.exists()
    assert not (tmp_path / "outputs" / "backtest_report.json").exists()
    assert not (tmp_path / "outputs" / "backtest_trades.jsonl").exists()
    assert (tmp_path / "state" / "trades.jsonl").read_text(encoding="utf-8") == before_state

    payload = json.loads(result.stdout)
    assert payload["output_dir"] == str(custom_dir)
    assert payload["report_path"] == str(report_path)
    assert payload["trades_path"] == str(trades_path)


def test_compare_strategies_csv_creates_report_and_leaves_state_untouched(tmp_path: Path, monkeypatch) -> None:
    _write_config(tmp_path)
    csv_path = _write_csv(tmp_path, with_gap=False)
    monkeypatch.chdir(tmp_path)

    before_trades = (tmp_path / "state" / "trades.jsonl").read_text(encoding="utf-8")
    before_hypotheses = (tmp_path / "state" / "hypotheses.jsonl").read_text(encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["compare-strategies-csv", str(csv_path)])

    assert result.exit_code == 0, result.output
    report_path = tmp_path / "outputs" / "strategy_comparison_report.json"
    assert report_path.exists()
    assert (tmp_path / "state" / "trades.jsonl").read_text(encoding="utf-8") == before_trades
    assert (tmp_path / "state" / "hypotheses.jsonl").read_text(encoding="utf-8") == before_hypotheses

    payload = json.loads(result.stdout)
    assert payload["command"] == "compare-strategies-csv"
    assert payload["rows"] == 50
    assert payload["gaps_detected"] == 0
    assert payload["report_path"] == "outputs/strategy_comparison_report.json"
    # Intentional: v0.31 adds donchian_breakout as Candidate 003
    assert {strategy["strategy_id"] for strategy in payload["strategies"]} == {
        "rsi_baseline",
        "ema_atr_trend",
        "donchian_breakout",
    }

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert {strategy["strategy_id"] for strategy in report["strategies"]} == {
        "rsi_baseline",
        "ema_atr_trend",
        "donchian_breakout",
    }


def test_compare_strategies_windows_csv_creates_report_and_leaves_state_untouched(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_config(tmp_path)
    csv_path = _write_csv(tmp_path, with_gap=False, rows=80)
    monkeypatch.chdir(tmp_path)

    before_trades = (tmp_path / "state" / "trades.jsonl").read_text(encoding="utf-8")
    before_hypotheses = (tmp_path / "state" / "hypotheses.jsonl").read_text(encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["compare-strategies-windows-csv", str(csv_path), "--windows", "4"])

    assert result.exit_code == 0, result.output
    report_path = tmp_path / "outputs" / "strategy_windows_comparison_report.json"
    assert report_path.exists()
    assert (tmp_path / "state" / "trades.jsonl").read_text(encoding="utf-8") == before_trades
    assert (tmp_path / "state" / "hypotheses.jsonl").read_text(encoding="utf-8") == before_hypotheses

    payload = json.loads(result.stdout)
    assert payload["command"] == "compare-strategies-windows-csv"
    assert payload["windows"] == 4
    assert payload["rows"] == 80
    assert payload["gaps_detected"] == 0
    assert payload["report_path"] == "outputs/strategy_windows_comparison_report.json"
    assert len(payload["results"]) == 4
    assert sum(window["rows"] for window in payload["results"]) == 80
    assert "best_strategy" not in payload
    # Intentional: v0.31 adds donchian_breakout as Candidate 003
    assert {summary["strategy_id"] for summary in payload["summary_by_strategy"]} == {
        "rsi_baseline",
        "ema_atr_trend",
        "donchian_breakout",
    }

    for window in payload["results"]:
        assert {strategy["strategy_id"] for strategy in window["strategies"]} == {
            "rsi_baseline",
            "ema_atr_trend",
            "donchian_breakout",
        }

    for summary in payload["summary_by_strategy"]:
        assert summary["windows"] == 4
        assert "total_trades" in summary
        assert "positive_expectancy_windows" in summary
        assert "average_expectancy" in summary
        assert "average_profit_factor" in summary

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(report["results"]) == 4
    assert "summary_by_strategy" in report
    assert "best_strategy" not in report
    assert {summary["strategy_id"] for summary in report["summary_by_strategy"]} == {
        "rsi_baseline",
        "ema_atr_trend",
        "donchian_breakout",
    }


def test_compare_strategies_windows_csv_rejects_too_few_windows(tmp_path: Path, monkeypatch) -> None:
    _write_config(tmp_path)
    csv_path = _write_csv(tmp_path, with_gap=False)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["compare-strategies-windows-csv", str(csv_path), "--windows", "1"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["command"] == "compare-strategies-windows-csv"
    assert payload["status"] == "failed"
    assert payload["reason"] == "windows_must_be_at_least_2"
    assert payload["windows"] == 1


def test_compare_strategies_windows_csv_rejects_more_windows_than_rows(tmp_path: Path, monkeypatch) -> None:
    _write_config(tmp_path)
    csv_path = _write_csv(tmp_path, with_gap=False)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["compare-strategies-windows-csv", str(csv_path), "--windows", "51"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["command"] == "compare-strategies-windows-csv"
    assert payload["status"] == "failed"
    assert payload["reason"] == "windows_must_not_exceed_rows"
    assert payload["windows"] == 51
    assert payload["rows"] == 50


def test_donchian_breakout_variant_atr_fields_match_exit(tmp_path: Path, monkeypatch) -> None:
    """The donchian_breakout variant built by cli.py must use the same atr_period and
    atr_stop_multiplier for risk sizing and for the exit trailing stop."""
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    base = load_strategy()
    variant = build_strategy_variant(base, "donchian_breakout")
    assert variant["exit"]["atr_period"] == variant["risk"]["atr_period"]
    assert variant["exit"]["atr_stop_multiplier"] == variant["risk"]["atr_stop_multiplier"]


def test_classify_strategy_summary_returns_research_status() -> None:
    research_policy = load_research_policy()

    assert classify_strategy_summary(
        {
            "total_trades": 9,
            "windows_with_trades": 4,
            "positive_expectancy_windows": 4,
            "average_expectancy": 1.0,
            "average_profit_factor": 2.0,
        },
        research_policy,
    ) == "insufficient_trades"
    assert classify_strategy_summary(
        {
            "total_trades": 10,
            "windows_with_trades": 2,
            "positive_expectancy_windows": 2,
            "average_expectancy": -0.1,
            "average_profit_factor": 1.5,
        },
        research_policy,
    ) == "weak"
    assert classify_strategy_summary(
        {
            "total_trades": 10,
            "windows_with_trades": 2,
            "positive_expectancy_windows": 1,
            "average_expectancy": 0.1,
            "average_profit_factor": 1.1,
        },
        research_policy,
    ) == "watchlist"
    assert classify_strategy_summary(
        {
            "total_trades": 10,
            "windows_with_trades": 2,
            "positive_expectancy_windows": 2,
            "average_expectancy": 0.1,
            "average_profit_factor": 1.1,
        },
        research_policy,
    ) == "candidate"


def test_show_comparison_report_displays_summary(tmp_path: Path, monkeypatch) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "outputs" / "strategy_windows_comparison_report.json"
    report_path.parent.mkdir()
    report_path.write_text(
        json.dumps(
            {
                "command": "compare-strategies-windows-csv",
                "csv_path": "data/BTCUSDT_1h.csv",
                "rows": 80,
                "windows": 4,
                "gaps_detected": 0,
                "report_path": "outputs/strategy_windows_comparison_report.json",
                "summary_by_strategy": [
                    {
                        "strategy_id": "rsi_baseline",
                        "windows": 4,
                        "windows_with_trades": 2,
                        "total_trades": 12,
                        "positive_expectancy_windows": 1,
                        "average_expectancy": 0.12,
                        "average_profit_factor": 1.35,
                    },
                    {
                        "strategy_id": "ema_atr_trend",
                        "windows": 4,
                        "windows_with_trades": 3,
                        "total_trades": 9,
                        "positive_expectancy_windows": 2,
                        "average_expectancy": 0.18,
                        "average_profit_factor": 1.48,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["show-comparison-report"])

    assert result.exit_code == 0
    assert "Comparison report" in result.stdout
    assert "Summary by strategy" in result.stdout
    assert "Acceptance thresholds" in result.stdout
    assert "Min total trades: 10" in result.stdout
    assert "rsi_baseline" in result.stdout
    assert "ema_atr_trend" in result.stdout
    assert "Research status: watchlist" in result.stdout
    assert "Research status: insufficient_trades" in result.stdout
    assert "Average expectancy" in result.stdout
    assert "Average profit factor" in result.stdout
    assert "best_strategy" not in result.stdout


def test_show_comparison_report_displays_all_research_statuses(tmp_path: Path, monkeypatch) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "outputs" / "strategy_windows_comparison_report.json"
    report_path.parent.mkdir()
    report_path.write_text(
        json.dumps(
            {
                "command": "compare-strategies-windows-csv",
                "csv_path": "data/BTCUSDT_1h.csv",
                "rows": 100,
                "windows": 4,
                "gaps_detected": 0,
                "summary_by_strategy": [
                    {
                        "strategy_id": "too_few",
                        "windows": 4,
                        "windows_with_trades": 1,
                        "total_trades": 12,
                        "positive_expectancy_windows": 2,
                        "average_expectancy": 0.2,
                        "average_profit_factor": 1.4,
                    },
                    {
                        "strategy_id": "weak_expectancy",
                        "windows": 4,
                        "windows_with_trades": 2,
                        "total_trades": 12,
                        "positive_expectancy_windows": 2,
                        "average_expectancy": -0.1,
                        "average_profit_factor": 1.4,
                    },
                    {
                        "strategy_id": "watch",
                        "windows": 4,
                        "windows_with_trades": 2,
                        "total_trades": 12,
                        "positive_expectancy_windows": 1,
                        "average_expectancy": 0.2,
                        "average_profit_factor": 1.4,
                    },
                    {
                        "strategy_id": "candidate",
                        "windows": 4,
                        "windows_with_trades": 2,
                        "total_trades": 12,
                        "positive_expectancy_windows": 2,
                        "average_expectancy": 0.2,
                        "average_profit_factor": 1.4,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["show-comparison-report"])

    assert result.exit_code == 0
    assert "Research status: insufficient_trades" in result.stdout
    assert "Research status: weak" in result.stdout
    assert "Research status: watchlist" in result.stdout
    assert "Research status: candidate" in result.stdout
    assert "best_strategy" not in result.stdout


def test_show_comparison_report_rejects_missing_file(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["show-comparison-report", "--path", str(tmp_path / "missing.json")])

    assert result.exit_code == 1
    assert "Comparison report not found" in result.stderr


def test_show_comparison_report_rejects_invalid_json(tmp_path: Path) -> None:
    report_path = tmp_path / "bad.json"
    report_path.write_text("{not-json", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["show-comparison-report", "--path", str(report_path)])

    assert result.exit_code == 1
    assert "Invalid comparison report JSON" in result.stderr


def test_show_comparison_report_rejects_missing_summary(tmp_path: Path) -> None:
    report_path = tmp_path / "missing-summary.json"
    report_path.write_text(json.dumps({"command": "compare-strategies-windows-csv"}), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["show-comparison-report", "--path", str(report_path)])

    assert result.exit_code == 1
    assert "Comparison report missing summary_by_strategy" in result.stderr


# --- get_classification_reasons tests ---

_POLICY = {
    "comparison_acceptance": {
        "min_total_trades": 10,
        "min_windows_with_trades": 2,
        "min_positive_expectancy_windows": 2,
        "min_average_expectancy": 0.0,
        "min_average_profit_factor": 1.1,
    }
}


def test_get_classification_reasons_insufficient_trades_too_few_trades() -> None:
    summary = {
        "total_trades": 5,
        "windows_with_trades": 3,
        "positive_expectancy_windows": 2,
        "average_expectancy": 0.5,
        "average_profit_factor": 1.5,
    }
    reasons = get_classification_reasons(summary, _POLICY, "insufficient_trades")
    assert "insufficient_total_trades" in reasons
    assert "insufficient_windows_with_trades" not in reasons


def test_get_classification_reasons_insufficient_trades_too_few_windows() -> None:
    summary = {
        "total_trades": 15,
        "windows_with_trades": 1,
        "positive_expectancy_windows": 2,
        "average_expectancy": 0.5,
        "average_profit_factor": 1.5,
    }
    reasons = get_classification_reasons(summary, _POLICY, "insufficient_trades")
    assert "insufficient_windows_with_trades" in reasons
    assert "insufficient_total_trades" not in reasons


def test_get_classification_reasons_insufficient_trades_both_reasons() -> None:
    summary = {
        "total_trades": 3,
        "windows_with_trades": 1,
        "positive_expectancy_windows": 0,
        "average_expectancy": 0.0,
        "average_profit_factor": 0.5,
    }
    reasons = get_classification_reasons(summary, _POLICY, "insufficient_trades")
    assert "insufficient_total_trades" in reasons
    assert "insufficient_windows_with_trades" in reasons


def test_get_classification_reasons_weak_expectancy() -> None:
    summary = {
        "total_trades": 10,
        "windows_with_trades": 2,
        "positive_expectancy_windows": 2,
        "average_expectancy": -0.1,
        "average_profit_factor": 1.5,
    }
    reasons = get_classification_reasons(summary, _POLICY, "weak")
    assert "average_expectancy_below_threshold" in reasons
    assert "average_profit_factor_below_threshold" not in reasons


def test_get_classification_reasons_weak_profit_factor() -> None:
    summary = {
        "total_trades": 10,
        "windows_with_trades": 2,
        "positive_expectancy_windows": 2,
        "average_expectancy": 0.5,
        "average_profit_factor": 0.9,
    }
    reasons = get_classification_reasons(summary, _POLICY, "weak")
    assert "average_profit_factor_below_threshold" in reasons
    assert "average_expectancy_below_threshold" not in reasons


def test_get_classification_reasons_weak_both() -> None:
    summary = {
        "total_trades": 10,
        "windows_with_trades": 2,
        "positive_expectancy_windows": 2,
        "average_expectancy": -0.5,
        "average_profit_factor": 0.8,
    }
    reasons = get_classification_reasons(summary, _POLICY, "weak")
    assert "average_expectancy_below_threshold" in reasons
    assert "average_profit_factor_below_threshold" in reasons


def test_get_classification_reasons_watchlist_contains_diagnostic_reasons() -> None:
    summary = {
        "total_trades": 12,
        "windows_with_trades": 3,
        "positive_expectancy_windows": 1,
        "average_expectancy": 0.2,
        "average_profit_factor": 1.4,
    }
    reasons = get_classification_reasons(summary, _POLICY, "watchlist")
    assert "insufficient_positive_expectancy_windows" in reasons
    assert "total_trades_above_threshold" in reasons
    assert "windows_with_trades_above_threshold" in reasons


def test_get_classification_reasons_candidate_contains_positive_reasons() -> None:
    summary = {
        "total_trades": 20,
        "windows_with_trades": 4,
        "positive_expectancy_windows": 3,
        "average_expectancy": 0.3,
        "average_profit_factor": 1.5,
    }
    reasons = get_classification_reasons(summary, _POLICY, "candidate")
    assert "total_trades_above_threshold" in reasons
    assert "windows_with_trades_above_threshold" in reasons
    assert "positive_expectancy_windows_above_threshold" in reasons
    assert "average_expectancy_above_threshold" in reasons
    assert "average_profit_factor_above_threshold" in reasons


def test_get_classification_reasons_uses_policy_thresholds() -> None:
    strict_policy = {
        "comparison_acceptance": {
            "min_total_trades": 50,
            "min_windows_with_trades": 5,
            "min_positive_expectancy_windows": 3,
            "min_average_expectancy": 0.0,
            "min_average_profit_factor": 1.1,
        }
    }
    summary = {
        "total_trades": 10,
        "windows_with_trades": 1,
        "positive_expectancy_windows": 2,
        "average_expectancy": 0.5,
        "average_profit_factor": 1.5,
    }
    reasons = get_classification_reasons(summary, strict_policy, "insufficient_trades")
    assert "insufficient_total_trades" in reasons
    assert "insufficient_windows_with_trades" in reasons


def test_show_comparison_report_displays_reasons(tmp_path: Path, monkeypatch) -> None:
    _write_research_policy(tmp_path)
    report_path = tmp_path / "outputs" / "strategy_windows_comparison_report.json"
    report_path.parent.mkdir()
    report_path.write_text(
        json.dumps(
            {
                "command": "compare-strategies-windows-csv",
                "csv_path": "data/BTCUSDT_1h.csv",
                "rows": 100,
                "windows": 4,
                "gaps_detected": 0,
                "summary_by_strategy": [
                    {
                        "strategy_id": "candidate_strat",
                        "windows": 4,
                        "windows_with_trades": 3,
                        "total_trades": 20,
                        "positive_expectancy_windows": 3,
                        "average_expectancy": 0.3,
                        "average_profit_factor": 1.5,
                    },
                    {
                        "strategy_id": "weak_strat",
                        "windows": 4,
                        "windows_with_trades": 2,
                        "total_trades": 12,
                        "positive_expectancy_windows": 2,
                        "average_expectancy": -0.2,
                        "average_profit_factor": 0.9,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["show-comparison-report"])

    assert result.exit_code == 0
    assert "Classification" not in result.stdout or "Research status" in result.stdout
    assert "Reasons:" in result.stdout
    assert "average_profit_factor_above_threshold" in result.stdout
    assert "average_expectancy_below_threshold" in result.stdout
    assert "best_strategy" not in result.stdout


def test_show_comparison_report_backward_compat_no_reasons_field(tmp_path: Path, monkeypatch) -> None:
    """Old reports without a 'reasons' field in JSON must still display correctly."""
    _write_research_policy(tmp_path)
    report_path = tmp_path / "outputs" / "strategy_windows_comparison_report.json"
    report_path.parent.mkdir()
    report_path.write_text(
        json.dumps(
            {
                "command": "compare-strategies-windows-csv",
                "csv_path": "data/BTCUSDT_1h.csv",
                "rows": 80,
                "windows": 4,
                "gaps_detected": 0,
                "summary_by_strategy": [
                    {
                        "strategy_id": "old_strategy",
                        "windows": 4,
                        "windows_with_trades": 1,
                        "total_trades": 5,
                        "positive_expectancy_windows": 0,
                        "average_expectancy": -0.1,
                        "average_profit_factor": 0.8,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["show-comparison-report"])

    assert result.exit_code == 0
    assert "old_strategy" in result.stdout
    assert "Research status: insufficient_trades" in result.stdout
    assert "Reasons:" in result.stdout
    assert "best_strategy" not in result.stdout
