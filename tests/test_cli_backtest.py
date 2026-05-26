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


def _write_csv(tmp_path: Path) -> Path:
    df = generate_sample_ohlcv(rows=50, seed=42)
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
    assert (tmp_path / "outputs" / "backtest_report.json").exists()
    assert (tmp_path / "outputs" / "backtest_trades.jsonl").exists()
    assert (tmp_path / "state" / "trades.jsonl").read_text(encoding="utf-8") == before_state

    report = json.loads((tmp_path / "outputs" / "backtest_report.json").read_text(encoding="utf-8"))
    assert report["rows"] == 50
    assert "score" in report
