from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from trading_agent.cli import app
from trading_agent.data import generate_sample_ohlcv


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
    assert {strategy["strategy_id"] for strategy in payload["strategies"]} == {
        "rsi_baseline",
        "ema_atr_trend",
    }

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert {strategy["strategy_id"] for strategy in report["strategies"]} == {
        "rsi_baseline",
        "ema_atr_trend",
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
    assert {summary["strategy_id"] for summary in payload["summary_by_strategy"]} == {
        "rsi_baseline",
        "ema_atr_trend",
    }

    for window in payload["results"]:
        assert {strategy["strategy_id"] for strategy in window["strategies"]} == {
            "rsi_baseline",
            "ema_atr_trend",
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
