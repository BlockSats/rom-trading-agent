from __future__ import annotations

from pathlib import Path

import pytest

from trading_agent.storage import append_jsonl, ensure_state_files, read_jsonl, save_strategy_version


def test_append_jsonl_writes_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    append_jsonl(path, {"a": 1})
    assert path.read_text(encoding="utf-8").strip() == '{"a": 1}'


def test_read_jsonl_returns_records(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    append_jsonl(path, {"a": 1})
    append_jsonl(path, {"b": 2})
    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_reading_missing_jsonl_returns_empty_list(tmp_path: Path) -> None:
    assert read_jsonl(tmp_path / "missing.jsonl") == []


def test_ensure_state_files_creates_required_state_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    ensure_state_files()
    assert Path("state/trades.jsonl").exists()
    assert Path("state/hypotheses.jsonl").exists()
    assert Path("state/heartbeat.json").exists()
    assert Path("state/history").exists()


def test_save_strategy_version_writes_versioned_yaml_file(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    history_dir = tmp_path / "state" / "history"
    config_dir.mkdir(parents=True)
    strategy_path = config_dir / "strategy.yaml"
    strategy_path.write_text(
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
  fee_pct: 0.1
  slippage_pct: 0.05
reflection:
  one_variable_only: true
  allowed_variables:
    - "entry.threshold"
""",
        encoding="utf-8",
    )
    saved = save_strategy_version(strategy_path=strategy_path, history_dir=history_dir)
    assert saved.name == "strategy_0001.yaml"
    assert saved.exists()


def test_save_strategy_version_does_not_silently_overwrite(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    history_dir = tmp_path / "state" / "history"
    config_dir.mkdir(parents=True)
    strategy_path = config_dir / "strategy.yaml"
    strategy_path.write_text(
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
  fee_pct: 0.1
  slippage_pct: 0.05
reflection:
  one_variable_only: true
  allowed_variables:
    - "entry.threshold"
""",
        encoding="utf-8",
    )
    save_strategy_version(strategy_path=strategy_path, history_dir=history_dir)
    with pytest.raises(FileExistsError):
        save_strategy_version(strategy_path=strategy_path, history_dir=history_dir)
