from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from typer.testing import CliRunner

from trading_agent.cli import app
from trading_agent.data import generate_sample_ohlcv


def _write_config(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "trades.jsonl").write_text('{"sentinel": true}\n', encoding="utf-8")

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


def _sample_binance_rows(rows: int = 60) -> list[dict[str, Any]]:
    df = generate_sample_ohlcv(rows=rows, seed=42)

    output: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        timestamp_ms = int(pd.Timestamp(row.timestamp).timestamp() * 1000)
        output.append(
            {
                "timestamp": timestamp_ms,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
            }
        )

    return output


def test_research_cycle_fetches_inspects_backtests_scores_and_leaves_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    before_state = (tmp_path / "state" / "trades.jsonl").read_text(encoding="utf-8")

    def fake_fetch_binance_ohlcv(symbol: str, interval: str, limit: int) -> list[dict[str, Any]]:
        assert symbol == "BTCUSDT"
        assert interval == "1h"
        assert limit == 60
        return _sample_binance_rows(rows=60)

    monkeypatch.setattr(
        "trading_agent.cli.fetch_binance_ohlcv",
        fake_fetch_binance_ohlcv,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "research-cycle",
            "--symbol",
            "BTCUSDT",
            "--interval",
            "1h",
            "--limit",
            "60",
            "--output",
            "data/BTCUSDT_1h.csv",
        ],
    )

    assert result.exit_code == 0, result.output

    payload = json.loads(result.stdout)

    assert payload["command"] == "research-cycle"
    assert payload["status"] == "completed"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["interval"] == "1h"
    assert payload["rows"] == 60
    assert payload["gaps_detected"] == 0
    assert "score" in payload
    assert payload["output_dir"] == "outputs"
    assert payload["research_report_path"] == "outputs/research_cycle_report.json"
    assert payload["backtest_report_path"] == "outputs/backtest_report.json"
    assert payload["trades_path"] == "outputs/backtest_trades.jsonl"

    assert (tmp_path / "data" / "BTCUSDT_1h.csv").exists()
    assert (tmp_path / "outputs" / "research_cycle_report.json").exists()
    assert (tmp_path / "outputs" / "backtest_report.json").exists()
    assert (tmp_path / "outputs" / "backtest_trades.jsonl").exists()

    after_state = (tmp_path / "state" / "trades.jsonl").read_text(encoding="utf-8")
    assert after_state == before_state


def test_research_cycle_writes_to_custom_output_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    before_state = (tmp_path / "state" / "trades.jsonl").read_text(encoding="utf-8")

    def fake_fetch_binance_ohlcv(symbol: str, interval: str, limit: int) -> list[dict[str, Any]]:
        assert symbol == "BTCUSDT"
        assert interval == "1h"
        assert limit == 60
        return _sample_binance_rows(rows=60)

    monkeypatch.setattr(
        "trading_agent.cli.fetch_binance_ohlcv",
        fake_fetch_binance_ohlcv,
    )

    custom_dir = tmp_path / "tmp" / "research-test"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "research-cycle",
            "--symbol",
            "BTCUSDT",
            "--interval",
            "1h",
            "--limit",
            "60",
            "--output",
            "data/BTCUSDT_1h.csv",
            "--output-dir",
            str(custom_dir),
        ],
    )

    assert result.exit_code == 0, result.output

    research_report_path = custom_dir / "research_cycle_report.json"
    backtest_report_path = custom_dir / "backtest_report.json"
    trades_path = custom_dir / "backtest_trades.jsonl"
    assert research_report_path.exists()
    assert backtest_report_path.exists()
    assert trades_path.exists()
    assert not (tmp_path / "outputs" / "research_cycle_report.json").exists()
    assert not (tmp_path / "outputs" / "backtest_report.json").exists()
    assert not (tmp_path / "outputs" / "backtest_trades.jsonl").exists()

    payload = json.loads(result.stdout)
    assert payload["output_dir"] == str(custom_dir)
    assert payload["research_report_path"] == str(research_report_path)
    assert payload["backtest_report_path"] == str(backtest_report_path)
    assert payload["trades_path"] == str(trades_path)

    after_state = (tmp_path / "state" / "trades.jsonl").read_text(encoding="utf-8")
    assert after_state == before_state


def test_research_cycle_with_reflect_writes_comparison_report_and_leaves_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    before_state = (tmp_path / "state" / "trades.jsonl").read_text(encoding="utf-8")
    before_strategy = (tmp_path / "config" / "strategy.yaml").read_text(encoding="utf-8")

    def fake_fetch_binance_ohlcv(symbol: str, interval: str, limit: int) -> list[dict[str, Any]]:
        assert symbol == "BTCUSDT"
        assert interval == "1h"
        assert limit == 60
        return _sample_binance_rows(rows=60)

    monkeypatch.setattr(
        "trading_agent.cli.fetch_binance_ohlcv",
        fake_fetch_binance_ohlcv,
    )

    runner = CliRunner()
    custom_dir = tmp_path / "tmp" / "research-reflect"
    result = runner.invoke(
        app,
        [
            "research-cycle",
            "--symbol",
            "BTCUSDT",
            "--interval",
            "1h",
            "--limit",
            "60",
            "--output",
            "data/BTCUSDT_1h.csv",
            "--output-dir",
            str(custom_dir),
            "--reflect",
        ],
    )

    assert result.exit_code == 0, result.output

    payload = json.loads(result.stdout)

    assert payload["command"] == "research-cycle"
    assert payload["status"] == "completed"
    assert payload["reflection"]["enabled"] is True
    assert payload["reflection"]["variable"] == "entry.threshold"
    assert payload["reflection"]["old_value"] == 30.0
    assert payload["reflection"]["new_value"] == 28.0

    reflection_report_path = custom_dir / "research_reflection_report.json"
    assert payload["reflection"]["report_path"] == str(reflection_report_path)
    assert reflection_report_path.exists()
    assert not (tmp_path / "outputs" / "research_reflection_report.json").exists()

    reflection_report = json.loads(reflection_report_path.read_text(encoding="utf-8"))

    assert reflection_report["command"] == "research-cycle"
    assert reflection_report["reflection_enabled"] is True
    assert reflection_report["status"] == "completed"
    assert reflection_report["hypothesis"]["variable"] == "entry.threshold"
    assert reflection_report["decision"] in {"keep", "reject"}
    assert "baseline" in reflection_report
    assert "candidate" in reflection_report

    after_state = (tmp_path / "state" / "trades.jsonl").read_text(encoding="utf-8")
    after_strategy = (tmp_path / "config" / "strategy.yaml").read_text(encoding="utf-8")

    assert after_state == before_state
    assert after_strategy == before_strategy
