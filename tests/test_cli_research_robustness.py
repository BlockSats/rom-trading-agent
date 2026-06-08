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


def _sample_binance_rows(rows: int = 80) -> list[dict[str, Any]]:
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


def test_research_robustness_rejects_too_few_windows() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "research-robustness",
            "--windows",
            "2",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)

    assert payload["command"] == "research-robustness"
    assert payload["status"] == "failed"
    assert payload["reason"] == "windows_must_be_at_least_3"


def test_research_robustness_writes_report_and_leaves_state(
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
        assert limit == 80
        return _sample_binance_rows(rows=80)

    monkeypatch.setattr(
        "trading_agent.cli.fetch_binance_ohlcv",
        fake_fetch_binance_ohlcv,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "research-robustness",
            "--symbol",
            "BTCUSDT",
            "--interval",
            "1h",
            "--limit",
            "80",
            "--output",
            "data/BTCUSDT_1h.csv",
            "--windows",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output

    payload = json.loads(result.stdout)

    assert payload["command"] == "research-robustness"
    assert payload["status"] == "completed"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["interval"] == "1h"
    assert payload["limit"] == 80
    assert payload["windows"] == 4
    assert payload["rows"] == 80
    assert payload["gaps_detected"] == 0
    assert payload["hypothesis"]["variable"] == "entry.threshold"
    assert payload["final_decision"] in {"keep", "reject"}
    assert len(payload["windows_results"]) == 4
    assert "summary" in payload
    assert payload["output_dir"] == "outputs"
    assert payload["robustness_report_path"] == "outputs/research_robustness_report.json"

    report_path = tmp_path / "outputs" / "research_robustness_report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["command"] == "research-robustness"
    assert report["windows"] == 4
    assert report["final_decision"] in {"keep", "reject"}

    after_state = (tmp_path / "state" / "trades.jsonl").read_text(encoding="utf-8")
    after_strategy = (tmp_path / "config" / "strategy.yaml").read_text(encoding="utf-8")

    assert after_state == before_state
    assert after_strategy == before_strategy


def test_research_robustness_writes_to_custom_output_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    def fake_fetch_binance_ohlcv(symbol: str, interval: str, limit: int) -> list[dict[str, Any]]:
        assert symbol == "BTCUSDT"
        assert interval == "1h"
        assert limit == 80
        return _sample_binance_rows(rows=80)

    monkeypatch.setattr(
        "trading_agent.cli.fetch_binance_ohlcv",
        fake_fetch_binance_ohlcv,
    )

    custom_dir = tmp_path / "tmp" / "robustness-test"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "research-robustness",
            "--symbol",
            "BTCUSDT",
            "--interval",
            "1h",
            "--limit",
            "80",
            "--output",
            "data/BTCUSDT_1h.csv",
            "--output-dir",
            str(custom_dir),
            "--windows",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output

    report_path = custom_dir / "research_robustness_report.json"
    assert report_path.exists()
    assert not (tmp_path / "outputs" / "research_robustness_report.json").exists()

    payload = json.loads(result.stdout)
    assert payload["output_dir"] == str(custom_dir)
    assert payload["robustness_report_path"] == str(report_path)
