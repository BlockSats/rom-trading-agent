"""Tests CLI de la commande archive-prospective-csv.

Tous les chemins utilisent tmp_path.
Aucun accès réseau. Aucun backtest, aucun scoring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from trading_agent.cli import app
from trading_agent.data import generate_sample_ohlcv
from trading_agent.storage import read_jsonl


runner = CliRunner()


def _write_sample_csv(path: Path, rows: int = 50, seed: int = 42) -> Path:
    df = generate_sample_ohlcv(rows=rows, seed=seed)
    df.to_csv(path, index=False)
    return path


def _write_invalid_csv(path: Path) -> Path:
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    return path


def _invoke(csv_path: Path, reserve_dir: Path, registry_path: Path, extra: list[str] | None = None) -> object:
    args = [
        "archive-prospective-csv",
        str(csv_path),
        "--asset", "BTCUSDT",
        "--timeframe", "1h",
        "--source", "binance-public",
        "--reserve-dir", str(reserve_dir),
        "--registry-path", str(registry_path),
    ]
    if extra:
        args.extend(extra)
    return runner.invoke(app, args)


# --- Test 1 : archivage réussi ---

def test_archive_success(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    result = _invoke(csv_path, tmp_path / "reserve", tmp_path / "registry.jsonl")
    assert result.exit_code == 0


# --- Test 2 : JSON de sortie lisible ---

def test_archive_json_output(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    result = _invoke(csv_path, tmp_path / "reserve", tmp_path / "registry.jsonl")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "archive-prospective-csv"
    assert payload["status"] == "completed"
    assert "registry_path" in payload
    assert "stored_path" in payload
    assert "batch" in payload
    batch = payload["batch"]
    assert batch["event_type"] == "prospective_batch_archived"
    assert batch["data_role"] == "prospective_holdout_candidate"
    assert batch["status"] == "accumulating"


# --- Test 3 : options obligatoires ---

def test_archive_missing_asset_fails(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    result = runner.invoke(app, [
        "archive-prospective-csv",
        str(csv_path),
        "--timeframe", "1h",
        "--source", "binance-public",
        "--reserve-dir", str(tmp_path / "reserve"),
        "--registry-path", str(tmp_path / "registry.jsonl"),
    ])
    assert result.exit_code != 0


def test_archive_missing_timeframe_fails(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    result = runner.invoke(app, [
        "archive-prospective-csv",
        str(csv_path),
        "--asset", "BTCUSDT",
        "--source", "binance-public",
        "--reserve-dir", str(tmp_path / "reserve"),
        "--registry-path", str(tmp_path / "registry.jsonl"),
    ])
    assert result.exit_code != 0


def test_archive_missing_source_fails(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    result = runner.invoke(app, [
        "archive-prospective-csv",
        str(csv_path),
        "--asset", "BTCUSDT",
        "--timeframe", "1h",
        "--reserve-dir", str(tmp_path / "reserve"),
        "--registry-path", str(tmp_path / "registry.jsonl"),
    ])
    assert result.exit_code != 0


# --- Test 4 : chemins personnalisés sous tmp_path ---

def test_archive_custom_paths(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    custom_reserve = tmp_path / "my_reserve"
    custom_registry = tmp_path / "my_state" / "my_registry.jsonl"

    result = runner.invoke(app, [
        "archive-prospective-csv",
        str(csv_path),
        "--asset", "ETHUSDT",
        "--timeframe", "4h",
        "--source", "kraken-public",
        "--reserve-dir", str(custom_reserve),
        "--registry-path", str(custom_registry),
    ])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert custom_reserve.exists()
    assert custom_registry.exists()
    records = read_jsonl(custom_registry)
    assert len(records) == 1


# --- Test 5 : doublon avec code de sortie non nul ---

def test_archive_duplicate_nonzero_exit(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    reserve_dir = tmp_path / "reserve"
    registry_path = tmp_path / "registry.jsonl"

    result1 = _invoke(csv_path, reserve_dir, registry_path)
    assert result1.exit_code == 0

    result2 = _invoke(csv_path, reserve_dir, registry_path)
    assert result2.exit_code != 0
    payload = json.loads(result2.stdout)
    assert payload["status"] == "failed"
    assert payload["reason"] == "duplicate_batch"


# --- Test 6 : CSV invalide avec code de sortie non nul ---

def test_archive_invalid_csv_nonzero_exit(tmp_path: Path) -> None:
    csv_path = _write_invalid_csv(tmp_path / "invalid.csv")
    result = _invoke(csv_path, tmp_path / "reserve", tmp_path / "registry.jsonl")
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "failed"
    assert payload["reason"] == "invalid_ohlcv"


# --- Test 7 : aucune sortie de backtest ou de scoring créée ---

def test_archive_no_backtest_output_created(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    result = _invoke(csv_path, tmp_path / "reserve", tmp_path / "registry.jsonl")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    batch = payload.get("batch", {})
    for field in ["winrate", "profit_factor", "expectancy", "score", "net_pnl", "strategy_id"]:
        assert field not in payload
        assert field not in batch


# --- Test 8 : aucun accès réseau ---

def test_archive_no_network_access(tmp_path: Path) -> None:
    """La commande doit se terminer sans erreur réseau dans un environnement isolé.

    L'absence d'import de fetch_binance_ohlcv ou de tout client HTTP dans
    prospective_reserve.py garantit structurellement l'absence d'accès réseau.
    Ce test vérifie que la commande s'exécute jusqu'au bout sans exception inattendue.
    """
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    result = _invoke(csv_path, tmp_path / "reserve", tmp_path / "registry.jsonl")
    assert result.exit_code == 0
    assert result.exception is None
