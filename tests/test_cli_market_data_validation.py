from __future__ import annotations

import json

from typer.testing import CliRunner

from trading_agent.cli import app


runner = CliRunner()


def fail_if_network_is_called(*_args, **_kwargs):
    raise AssertionError("fetch_binance_ohlcv should not be called for invalid limits")


def assert_invalid_limit_result(result, expected_command: str, expected_limit: int) -> None:
    assert result.exit_code == 1

    payload = json.loads(result.stdout)

    assert payload["command"] == expected_command
    assert payload["status"] == "failed"
    assert payload["reason"] == "limit_must_be_between_1_and_1000"
    assert payload["limit"] == expected_limit


def test_fetch_ohlcv_rejects_limit_below_1(monkeypatch) -> None:
    monkeypatch.setattr("trading_agent.cli.fetch_binance_ohlcv", fail_if_network_is_called)

    result = runner.invoke(app, ["fetch-ohlcv", "--limit", "0"])

    assert_invalid_limit_result(result, "fetch-ohlcv", 0)


def test_fetch_ohlcv_rejects_limit_above_1000(monkeypatch) -> None:
    monkeypatch.setattr("trading_agent.cli.fetch_binance_ohlcv", fail_if_network_is_called)

    result = runner.invoke(app, ["fetch-ohlcv", "--limit", "1001"])

    assert_invalid_limit_result(result, "fetch-ohlcv", 1001)


def test_research_cycle_rejects_invalid_limit(monkeypatch) -> None:
    monkeypatch.setattr("trading_agent.cli.fetch_binance_ohlcv", fail_if_network_is_called)

    result = runner.invoke(app, ["research-cycle", "--limit", "0"])

    assert_invalid_limit_result(result, "research-cycle", 0)


def test_research_robustness_rejects_invalid_limit(monkeypatch) -> None:
    monkeypatch.setattr("trading_agent.cli.fetch_binance_ohlcv", fail_if_network_is_called)

    result = runner.invoke(app, ["research-robustness", "--limit", "1001"])

    assert_invalid_limit_result(result, "research-robustness", 1001)
