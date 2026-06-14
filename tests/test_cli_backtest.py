from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from trading_agent.cli import (
    app,
    build_strategy_variant,
    classify_strategy_summary,
    compute_window_robustness_diagnostics,
    get_classification_reasons,
)
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


# --- compute_window_robustness_diagnostics unit tests ---


def test_compute_window_robustness_diagnostics_basic() -> None:
    summary = {
        "windows": 4,
        "windows_with_trades": 3,
        "total_trades": 12,
        "positive_expectancy_windows": 2,
    }
    window_results = [
        {"expectancy": 0.5, "profit_factor": 1.5},
        {"expectancy": -0.2, "profit_factor": 0.8},
        {"expectancy": 0.3, "profit_factor": 1.2},
        {"expectancy": 0.0, "profit_factor": None},
    ]
    d = compute_window_robustness_diagnostics(summary, window_results)
    assert d["zero_trade_windows"] == 1
    assert d["negative_expectancy_windows"] == 1
    assert d["average_trades_per_active_window"] == pytest.approx(4.0)
    assert d["window_participation_rate"] == pytest.approx(0.75)
    assert d["positive_expectancy_rate"] == pytest.approx(2 / 3)
    assert d["expectancy_min"] == pytest.approx(-0.2)
    assert d["expectancy_max"] == pytest.approx(0.5)
    assert d["profit_factor_min"] == pytest.approx(0.8)
    assert d["profit_factor_max"] == pytest.approx(1.5)


def test_compute_window_robustness_diagnostics_profit_factor_none_filtered() -> None:
    summary = {
        "windows": 3,
        "windows_with_trades": 2,
        "total_trades": 6,
        "positive_expectancy_windows": 1,
    }
    window_results = [
        {"expectancy": 0.5, "profit_factor": None},
        {"expectancy": 0.2, "profit_factor": 1.3},
        {"expectancy": 0.0, "profit_factor": None},
    ]
    d = compute_window_robustness_diagnostics(summary, window_results)
    # Only 1.3 survives the None filter
    assert d["profit_factor_min"] == pytest.approx(1.3)
    assert d["profit_factor_max"] == pytest.approx(1.3)


def test_compute_window_robustness_diagnostics_all_profit_factor_none() -> None:
    summary = {
        "windows": 3,
        "windows_with_trades": 0,
        "total_trades": 0,
        "positive_expectancy_windows": 0,
    }
    window_results = [
        {"expectancy": 0.0, "profit_factor": None},
        {"expectancy": 0.0, "profit_factor": None},
        {"expectancy": 0.0, "profit_factor": None},
    ]
    d = compute_window_robustness_diagnostics(summary, window_results)
    assert d["profit_factor_min"] is None
    assert d["profit_factor_max"] is None
    assert d["negative_expectancy_windows"] == 0


def test_compute_window_robustness_diagnostics_zero_windows_with_trades() -> None:
    summary = {
        "windows": 4,
        "windows_with_trades": 0,
        "total_trades": 0,
        "positive_expectancy_windows": 0,
    }
    d = compute_window_robustness_diagnostics(summary, [])
    assert d["zero_trade_windows"] == 4
    assert d["average_trades_per_active_window"] is None
    assert d["positive_expectancy_rate"] is None
    assert d["window_participation_rate"] == pytest.approx(0.0)


def test_compute_window_robustness_diagnostics_expectancy_zero_is_neutral() -> None:
    summary = {
        "windows": 3,
        "windows_with_trades": 2,
        "total_trades": 5,
        "positive_expectancy_windows": 0,
    }
    window_results = [
        {"expectancy": 0.0, "profit_factor": None},
        {"expectancy": 0.0, "profit_factor": None},
        {"expectancy": 0.0, "profit_factor": None},
    ]
    d = compute_window_robustness_diagnostics(summary, window_results)
    # expectancy == 0 is neutral: not negative
    assert d["negative_expectancy_windows"] == 0


def test_compute_window_robustness_diagnostics_no_window_results() -> None:
    summary = {
        "windows": 4,
        "windows_with_trades": 2,
        "total_trades": 8,
        "positive_expectancy_windows": 1,
    }
    d = compute_window_robustness_diagnostics(summary, [])
    assert d["zero_trade_windows"] == 2
    assert d["negative_expectancy_windows"] is None
    assert d["expectancy_min"] is None
    assert d["expectancy_max"] is None
    assert d["profit_factor_min"] is None
    assert d["profit_factor_max"] is None
    assert d["average_trades_per_active_window"] == pytest.approx(4.0)
    assert d["window_participation_rate"] == pytest.approx(0.5)
    assert d["positive_expectancy_rate"] == pytest.approx(0.5)


# --- show-comparison-report window diagnostics CLI tests ---


def test_show_comparison_report_displays_window_diagnostics(tmp_path: Path, monkeypatch) -> None:
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
                "results": [
                    {"window": 1, "strategies": [{"strategy_id": "rsi_baseline", "closed_trades": 3, "expectancy": 0.5, "profit_factor": 1.4}]},
                    {"window": 2, "strategies": [{"strategy_id": "rsi_baseline", "closed_trades": 0, "expectancy": 0.0, "profit_factor": None}]},
                    {"window": 3, "strategies": [{"strategy_id": "rsi_baseline", "closed_trades": 4, "expectancy": -0.2, "profit_factor": 0.9}]},
                    {"window": 4, "strategies": [{"strategy_id": "rsi_baseline", "closed_trades": 2, "expectancy": 0.3, "profit_factor": 1.1}]},
                ],
                "summary_by_strategy": [
                    {
                        "strategy_id": "rsi_baseline",
                        "windows": 4,
                        "windows_with_trades": 3,
                        "total_trades": 9,
                        "positive_expectancy_windows": 2,
                        "average_expectancy": 0.15,
                        "average_profit_factor": 1.13,
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
    assert "Window robustness diagnostics" in result.stdout
    assert "Zero trade windows: 1" in result.stdout
    assert "Window participation rate:" in result.stdout
    assert "Average trades per active window:" in result.stdout
    assert "Positive expectancy rate:" in result.stdout
    assert "Negative expectancy windows: 1" in result.stdout
    assert "Profit factor min:" in result.stdout
    assert "Profit factor max:" in result.stdout
    assert "best_strategy" not in result.stdout


def test_show_comparison_report_backward_compat_no_results_key(tmp_path: Path, monkeypatch) -> None:
    """Old reports without a 'results' key still display partial diagnostics from summary data."""
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
                        "windows_with_trades": 2,
                        "total_trades": 6,
                        "positive_expectancy_windows": 1,
                        "average_expectancy": 0.1,
                        "average_profit_factor": 1.2,
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
    # zero_trade_windows is computable from summary data even without 'results'
    assert "Zero trade windows: 2" in result.stdout
    # per-window stats require 'results' — must be absent for old reports
    assert "Negative expectancy windows:" not in result.stdout


def test_show_comparison_report_no_best_strategy_in_diagnostics(tmp_path: Path, monkeypatch) -> None:
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
                        "strategy_id": "s1",
                        "windows": 4,
                        "windows_with_trades": 4,
                        "total_trades": 20,
                        "positive_expectancy_windows": 4,
                        "average_expectancy": 0.5,
                        "average_profit_factor": 1.5,
                    },
                    {
                        "strategy_id": "s2",
                        "windows": 4,
                        "windows_with_trades": 4,
                        "total_trades": 15,
                        "positive_expectancy_windows": 3,
                        "average_expectancy": 0.3,
                        "average_profit_factor": 1.3,
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
    assert "best_strategy" not in result.stdout


def test_show_comparison_report_no_automatic_sorting(tmp_path: Path, monkeypatch) -> None:
    """Output order must match report order, not be sorted by any metric."""
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
                        "strategy_id": "zeta_strategy",
                        "windows": 4,
                        "windows_with_trades": 1,
                        "total_trades": 3,
                        "positive_expectancy_windows": 1,
                        "average_expectancy": 0.1,
                        "average_profit_factor": 1.1,
                    },
                    {
                        "strategy_id": "alpha_strategy",
                        "windows": 4,
                        "windows_with_trades": 4,
                        "total_trades": 20,
                        "positive_expectancy_windows": 4,
                        "average_expectancy": 0.5,
                        "average_profit_factor": 1.5,
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
    # zeta_strategy appears before alpha_strategy — order preserved from the report
    assert result.stdout.index("zeta_strategy") < result.stdout.index("alpha_strategy")


def test_show_comparison_report_does_not_recalculate_backtest(tmp_path: Path, monkeypatch) -> None:
    """show-comparison-report must never load a CSV or run a backtest."""
    _write_research_policy(tmp_path)
    report_path = tmp_path / "outputs" / "strategy_windows_comparison_report.json"
    report_path.parent.mkdir()
    report_path.write_text(
        json.dumps(
            {
                "command": "compare-strategies-windows-csv",
                "csv_path": "data/nonexistent.csv",
                "rows": 100,
                "windows": 4,
                "gaps_detected": 0,
                "summary_by_strategy": [
                    {
                        "strategy_id": "strat_a",
                        "windows": 4,
                        "windows_with_trades": 2,
                        "total_trades": 8,
                        "positive_expectancy_windows": 2,
                        "average_expectancy": 0.2,
                        "average_profit_factor": 1.2,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["show-comparison-report"])

    # If it tried to load nonexistent.csv it would fail — it must succeed without it
    assert result.exit_code == 0
    assert "strat_a" in result.stdout


def test_show_comparison_report_many_zero_trade_windows(tmp_path: Path, monkeypatch) -> None:
    """A strategy with mostly inactive windows shows zero_trade_windows alongside participation rate."""
    _write_research_policy(tmp_path)
    report_path = tmp_path / "outputs" / "strategy_windows_comparison_report.json"
    report_path.parent.mkdir()
    report_path.write_text(
        json.dumps(
            {
                "command": "compare-strategies-windows-csv",
                "csv_path": "data/BTCUSDT_1h.csv",
                "rows": 80,
                "windows": 8,
                "gaps_detected": 0,
                "summary_by_strategy": [
                    {
                        "strategy_id": "inactive_strategy",
                        "windows": 8,
                        "windows_with_trades": 1,
                        "total_trades": 15,
                        "positive_expectancy_windows": 1,
                        "average_expectancy": 0.5,
                        "average_profit_factor": 2.0,
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
    assert "inactive_strategy" in result.stdout
    # 8 - 1 = 7 inactive windows must be visible
    assert "Zero trade windows: 7" in result.stdout
    assert "best_strategy" not in result.stdout


def test_compare_strategies_assets_csv_creates_report_and_leaves_state_untouched(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_config(tmp_path)
    btc_csv = tmp_path / "btc.csv"
    eth_csv = tmp_path / "eth.csv"
    generate_sample_ohlcv(rows=80, seed=42).to_csv(btc_csv, index=False)
    generate_sample_ohlcv(rows=80, seed=99).to_csv(eth_csv, index=False)
    monkeypatch.chdir(tmp_path)

    before_trades = (tmp_path / "state" / "trades.jsonl").read_text(encoding="utf-8")
    before_hypotheses = (tmp_path / "state" / "hypotheses.jsonl").read_text(encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, [
        "compare-strategies-assets-csv",
        "--asset", f"BTCUSDT:{btc_csv}",
        "--asset", f"ETHUSDT:{eth_csv}",
        "--windows", "4",
    ])

    assert result.exit_code == 0, result.output
    report_path = tmp_path / "outputs" / "strategy_assets_comparison_report.json"
    assert report_path.exists()
    assert (tmp_path / "state" / "trades.jsonl").read_text(encoding="utf-8") == before_trades
    assert (tmp_path / "state" / "hypotheses.jsonl").read_text(encoding="utf-8") == before_hypotheses

    payload = json.loads(result.stdout)
    assert payload["command"] == "compare-strategies-assets-csv"
    assert payload["windows"] == 4
    assert payload["report_path"] == "outputs/strategy_assets_comparison_report.json"
    assert payload["output_dir"] == "outputs"
    assert "best_strategy" not in payload
    assert "best_asset" not in payload

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "results_by_asset" in report
    assert "best_strategy" not in report
    assert "best_asset" not in report


def test_compare_strategies_assets_csv_rejects_invalid_asset_format(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, [
        "compare-strategies-assets-csv",
        "--asset", "BTCUSDT_no_colon.csv",
        "--windows", "4",
    ])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["command"] == "compare-strategies-assets-csv"
    assert payload["status"] == "failed"
    assert payload["reason"] == "invalid_asset_format"
    assert payload["asset"] == "BTCUSDT_no_colon.csv"


def test_compare_strategies_assets_csv_rejects_empty_symbol(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, [
        "compare-strategies-assets-csv",
        "--asset", ":data/file.csv",
        "--windows", "4",
    ])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["command"] == "compare-strategies-assets-csv"
    assert payload["status"] == "failed"
    assert payload["reason"] == "empty_symbol"


def test_compare_strategies_assets_csv_rejects_empty_csv_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, [
        "compare-strategies-assets-csv",
        "--asset", "BTCUSDT:",
        "--windows", "4",
    ])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["command"] == "compare-strategies-assets-csv"
    assert payload["status"] == "failed"
    assert payload["reason"] == "empty_csv_path"


def test_compare_strategies_assets_csv_rejects_too_few_windows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_config(tmp_path)
    csv_path = _write_csv(tmp_path, with_gap=False, rows=80)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, [
        "compare-strategies-assets-csv",
        "--asset", f"BTCUSDT:{csv_path}",
        "--windows", "1",
    ])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["command"] == "compare-strategies-assets-csv"
    assert payload["status"] == "failed"
    assert payload["reason"] == "windows_must_be_at_least_2"
    assert payload["windows"] == 1


def test_compare_strategies_assets_csv_rejects_more_windows_than_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_config(tmp_path)
    csv_path = _write_csv(tmp_path, with_gap=False, rows=10)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, [
        "compare-strategies-assets-csv",
        "--asset", f"BTCUSDT:{csv_path}",
        "--windows", "20",
    ])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["command"] == "compare-strategies-assets-csv"
    assert payload["status"] == "failed"
    assert payload["reason"] == "windows_must_not_exceed_rows"
    assert payload["windows"] == 20
    assert payload["rows"] == 10
    assert payload["symbol"] == "BTCUSDT"


def test_compare_strategies_assets_csv_results_by_asset_has_correct_symbols(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_config(tmp_path)
    btc_csv = tmp_path / "btc.csv"
    eth_csv = tmp_path / "eth.csv"
    generate_sample_ohlcv(rows=80, seed=42).to_csv(btc_csv, index=False)
    generate_sample_ohlcv(rows=80, seed=99).to_csv(eth_csv, index=False)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, [
        "compare-strategies-assets-csv",
        "--asset", f"BTCUSDT:{btc_csv}",
        "--asset", f"ETHUSDT:{eth_csv}",
        "--windows", "4",
    ])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert set(payload["results_by_asset"].keys()) == {"BTCUSDT", "ETHUSDT"}
    assert payload["assets"] == ["BTCUSDT", "ETHUSDT"]


def test_compare_strategies_assets_csv_strategies_in_fixed_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_config(tmp_path)
    csv_path = _write_csv(tmp_path, with_gap=False, rows=80)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, [
        "compare-strategies-assets-csv",
        "--asset", f"BTCUSDT:{csv_path}",
        "--windows", "4",
    ])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["strategies"] == ["rsi_baseline", "ema_atr_trend", "donchian_breakout"]


def test_compare_strategies_assets_csv_has_no_forbidden_keys(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_config(tmp_path)
    csv_path = _write_csv(tmp_path, with_gap=False, rows=80)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, [
        "compare-strategies-assets-csv",
        "--asset", f"BTCUSDT:{csv_path}",
        "--windows", "4",
    ])

    assert result.exit_code == 0, result.output
    report_text = json.dumps(json.loads(result.stdout))
    for forbidden in [
        "best_strategy",
        "best_asset",
        "winner",
        "global_score",
        "selected_strategy",
    ]:
        assert forbidden not in report_text, f"Forbidden key found: {forbidden}"


# ---------------------------------------------------------------------------
# show-walk-forward-report CLI tests
# ---------------------------------------------------------------------------

_SAMPLE_WF_REPORT = {
    "walk_forward": {"train_window": 60, "test_window": 20, "step": 20, "windows": 3},
    "strategy": {"strategy_id": "rsi_baseline", "asset": "BTC/USDT", "timeframe": "1h"},
    "summary": {"windows": 3, "windows_with_trades": 2, "total_trades": 7},
    "results": [
        {
            "window_index": 0,
            "train_start": "2024-01-01 00:00:00",
            "train_end": "2024-03-01 00:00:00",
            "test_start": "2024-03-01 00:00:00",
            "test_end": "2024-04-01 00:00:00",
            "test_rows": 20,
            "summary": {
                "closed_trades": 3,
                "total_net_pnl": 150.0,
                "total_return": 0.015,
                "max_drawdown": 0.05,
                "profit_factor": 1.5,
                "expectancy": 50.0,
            },
        },
        {
            "window_index": 1,
            "train_start": "2024-02-01 00:00:00",
            "train_end": "2024-04-01 00:00:00",
            "test_start": "2024-04-01 00:00:00",
            "test_end": "2024-05-01 00:00:00",
            "test_rows": 20,
            "summary": {
                "closed_trades": 4,
                "total_net_pnl": -30.0,
                "total_return": -0.003,
                "max_drawdown": 0.02,
                "profit_factor": 0.8,
                "expectancy": -7.5,
            },
        },
    ],
}


def _write_wf_report(tmp_path: Path, report: dict | None = None) -> Path:
    report_path = tmp_path / "outputs" / "walk_forward_report.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report or _SAMPLE_WF_REPORT), encoding="utf-8")
    return report_path


def test_show_walk_forward_report_default_path(tmp_path: Path, monkeypatch) -> None:
    report_path = _write_wf_report(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["show-walk-forward-report"])
    assert result.exit_code == 0, result.output
    assert "Walk-forward report" in result.output


def test_show_walk_forward_report_explicit_path(tmp_path: Path) -> None:
    report_path = _write_wf_report(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    assert result.exit_code == 0, result.output
    assert "Walk-forward report" in result.output


def test_show_walk_forward_report_displays_walk_forward_section(tmp_path: Path) -> None:
    report_path = _write_wf_report(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    assert result.exit_code == 0, result.output
    assert "Train window: 60" in result.output
    assert "Test window: 20" in result.output
    assert "Step: 20" in result.output
    assert "Windows: 3" in result.output


def test_show_walk_forward_report_displays_strategy_section(tmp_path: Path) -> None:
    report_path = _write_wf_report(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    assert result.exit_code == 0, result.output
    assert "Strategy ID: rsi_baseline" in result.output
    assert "Asset: BTC/USDT" in result.output
    assert "Timeframe: 1h" in result.output


def test_show_walk_forward_report_displays_summary_section(tmp_path: Path) -> None:
    report_path = _write_wf_report(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    assert result.exit_code == 0, result.output
    assert "Windows with trades: 2" in result.output
    assert "Total trades: 7" in result.output


def test_show_walk_forward_report_displays_results(tmp_path: Path) -> None:
    report_path = _write_wf_report(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    assert result.exit_code == 0, result.output
    assert "Window 0" in result.output
    assert "Window 1" in result.output
    assert "Test start:" in result.output
    assert "Test end:" in result.output
    assert "Test rows: 20" in result.output
    assert "Closed trades:" in result.output
    assert "Total net PnL:" in result.output
    assert "Total return:" in result.output
    assert "Max drawdown:" in result.output
    assert "Profit factor:" in result.output
    assert "Expectancy:" in result.output


def test_show_walk_forward_report_partial_report_accepted(tmp_path: Path) -> None:
    partial = {"walk_forward": {"train_window": 60, "test_window": 20}}
    report_path = _write_wf_report(tmp_path, partial)
    runner = CliRunner()
    result = runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    assert result.exit_code == 0, result.output
    assert "Train window: 60" in result.output


def test_show_walk_forward_report_missing_file(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["show-walk-forward-report", "--path", str(tmp_path / "missing.json")])
    assert result.exit_code == 1
    assert "not found" in result.output.lower() or "not found" in (result.stderr or "").lower()


def test_show_walk_forward_report_invalid_json(tmp_path: Path) -> None:
    report_path = tmp_path / "bad.json"
    report_path.write_text("not valid json {{{", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    assert result.exit_code == 1


def test_show_walk_forward_report_non_dict_root(tmp_path: Path) -> None:
    report_path = tmp_path / "list.json"
    report_path.write_text("[1, 2, 3]", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    assert result.exit_code == 1


def test_show_walk_forward_report_no_recalculation(tmp_path: Path, monkeypatch) -> None:
    import trading_agent.cli as cli_module
    called = []

    def fake_run_backtest(*args, **kwargs):
        called.append(True)
        return {"trades": [], "initial_balance": 1000.0}

    monkeypatch.setattr(cli_module, "run_backtest", fake_run_backtest, raising=False)
    report_path = _write_wf_report(tmp_path)
    runner = CliRunner()
    runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    assert called == [], "run_backtest should not be called by show-walk-forward-report"


def test_show_walk_forward_report_no_write_to_outputs(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    report_path = outputs_dir / "walk_forward_report.json"
    report_path.write_text(json.dumps(_SAMPLE_WF_REPORT), encoding="utf-8")
    before = set(outputs_dir.iterdir())
    runner = CliRunner()
    runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    after = set(outputs_dir.iterdir())
    assert before == after


def test_show_walk_forward_report_no_write_to_state(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    before = set(state_dir.iterdir())
    report_path = _write_wf_report(tmp_path)
    runner = CliRunner()
    runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    after = set(state_dir.iterdir())
    assert before == after


def test_show_walk_forward_report_no_forbidden_keys(tmp_path: Path) -> None:
    report_path = _write_wf_report(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    assert result.exit_code == 0, result.output
    for forbidden in ["best_strategy", "best_asset", "winner", "rank", "ranking",
                      "global_score", "selected_strategy"]:
        assert forbidden not in result.output, f"Forbidden key found in output: {forbidden}"


def test_show_wf_report_walk_forward_non_dict(tmp_path: Path) -> None:
    report = {"walk_forward": "not_a_dict", "strategy": {}, "summary": {}, "results": []}
    report_path = _write_wf_report(tmp_path, report)
    runner = CliRunner()
    result = runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    assert result.exit_code == 0, result.output


def test_show_wf_report_strategy_or_summary_non_dict(tmp_path: Path) -> None:
    report = {"walk_forward": {}, "strategy": 42, "summary": "bad", "results": []}
    report_path = _write_wf_report(tmp_path, report)
    runner = CliRunner()
    result = runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    assert result.exit_code == 0, result.output


def test_show_wf_report_results_entry_non_dict(tmp_path: Path) -> None:
    valid_entry = {
        "window_index": 0,
        "test_start": "2024-03-01",
        "test_end": "2024-04-01",
        "test_rows": 10,
        "summary": {"closed_trades": 2},
    }
    report = {"results": ["bad_entry", valid_entry]}
    report_path = _write_wf_report(tmp_path, report)
    runner = CliRunner()
    result = runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    assert result.exit_code == 0, result.output
    assert "Window 0" in result.output


def test_show_wf_report_window_summary_non_dict(tmp_path: Path) -> None:
    report = {
        "results": [
            {"window_index": 5, "test_start": "2024-03-01", "summary": "oops"},
        ]
    }
    report_path = _write_wf_report(tmp_path, report)
    runner = CliRunner()
    result = runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    assert result.exit_code == 0, result.output
    assert "Window 5" in result.output


# ---------------------------------------------------------------------------
# Tests v0.47 — validation_context dans le viewer CLI
# ---------------------------------------------------------------------------

_SAMPLE_WF_REPORT_WITH_VC = {
    **_SAMPLE_WF_REPORT,
    "validation_context": {
        "mode": "exploratory_walk_forward",
        "data_role": "exploratory_data",
        "confirmatory_holdout_used": False,
        "prospective_holdout_used": False,
        "paper_forward_used": False,
        "parameter_optimization_performed": False,
        "selection_performed": False,
    },
}


def test_show_wf_report_displays_validation_context(tmp_path: Path) -> None:
    report_path = _write_wf_report(tmp_path, _SAMPLE_WF_REPORT_WITH_VC)
    runner = CliRunner()
    result = runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    assert result.exit_code == 0, result.output
    assert "Validation context" in result.output
    assert "Mode: exploratory_walk_forward" in result.output
    assert "Data role: exploratory_data" in result.output
    assert "Confirmatory holdout used: False" in result.output
    assert "Paper-forward used: False" in result.output
    assert "Parameter optimization performed: False" in result.output
    assert "Selection performed: False" in result.output


def test_show_wf_report_legacy_report_readable(tmp_path: Path) -> None:
    """Un ancien rapport sans validation_context reste lisible sans erreur."""
    report_path = _write_wf_report(tmp_path, _SAMPLE_WF_REPORT)
    runner = CliRunner()
    result = runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    assert result.exit_code == 0, result.output
    assert "Walk-forward report" in result.output


def test_show_wf_report_no_write_with_validation_context(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    report_path = outputs_dir / "walk_forward_report.json"
    report_path.write_text(
        json.dumps(_SAMPLE_WF_REPORT_WITH_VC), encoding="utf-8"
    )
    before = set(outputs_dir.iterdir())
    runner = CliRunner()
    runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    after = set(outputs_dir.iterdir())
    assert before == after


def test_show_wf_report_no_forbidden_keys_in_output_with_vc(tmp_path: Path) -> None:
    report_path = _write_wf_report(tmp_path, _SAMPLE_WF_REPORT_WITH_VC)
    runner = CliRunner()
    result = runner.invoke(app, ["show-walk-forward-report", "--path", str(report_path)])
    assert result.exit_code == 0, result.output
    forbidden = {
        "best_strategy", "best_asset", "winner", "rank", "ranking",
        "global_score", "selected_strategy", "promotion", "auto_selection",
    }
    for key in forbidden:
        assert key not in result.output, f"Champ interdit trouvé dans l'output : {key}"
