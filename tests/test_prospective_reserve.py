"""Tests du module prospective_reserve.

Toutes les opérations de fichier utilisent tmp_path.
Aucun accès réseau. Aucun backtest, aucune stratégie.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trading_agent.data import generate_sample_ohlcv
from trading_agent.prospective_reserve import (
    DuplicateBatchError,
    InvalidMetadataError,
    InvalidOHLCVError,
    ReserveFileExistsError,
    archive_prospective_csv,
    build_batch_record,
    compute_sha256,
    validate_text_metadata,
)
from trading_agent.storage import read_jsonl


FORBIDDEN_PERFORMANCE_FIELDS = {
    "strategy_id", "return", "total_return", "pnl", "net_pnl", "score",
    "winrate", "profit_factor", "expectancy", "sharpe", "best_strategy",
    "winner", "rank", "selected_strategy",
}

FIXED_TS = "2026-01-15T10:00:00+00:00"


def _write_sample_csv(path: Path, rows: int = 50, seed: int = 42) -> Path:
    df = generate_sample_ohlcv(rows=rows, seed=seed)
    df.to_csv(path, index=False)
    return path


def _write_invalid_csv(path: Path) -> Path:
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    return path


def _write_csv_with_gap(path: Path) -> Path:
    df = generate_sample_ohlcv(rows=50, seed=42)
    df = pd.concat([df.iloc[:20], df.iloc[25:]]).reset_index(drop=True)
    df.to_csv(path, index=False)
    return path


def _archive(tmp_path: Path, csv_path: Path, seed: int = 42) -> dict:
    return archive_prospective_csv(
        path=csv_path,
        asset="BTCUSDT",
        timeframe="1h",
        source="binance-public",
        reserve_dir=tmp_path / "reserve",
        registry_path=tmp_path / "registry.jsonl",
        registered_at=FIXED_TS,
    )


# --- Test 1 : SHA-256 déterministe ---

def test_compute_sha256_deterministic(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    sha1 = compute_sha256(csv_path)
    sha2 = compute_sha256(csv_path)
    assert sha1 == sha2
    assert len(sha1) == 64


# --- Test 2 : hash différent lorsque les octets diffèrent ---

def test_compute_sha256_differs_for_different_bytes(tmp_path: Path) -> None:
    csv1 = _write_sample_csv(tmp_path / "ohlcv1.csv", seed=42)
    csv2 = _write_sample_csv(tmp_path / "ohlcv2.csv", seed=99)
    assert compute_sha256(csv1) != compute_sha256(csv2)


# --- Test 3 : métadonnées vides refusées ---

@pytest.mark.parametrize("asset,timeframe,source", [
    ("", "1h", "binance"),
    ("BTCUSDT", "", "binance"),
    ("BTCUSDT", "1h", ""),
    ("  ", "1h", "binance"),
    ("BTCUSDT", "  ", "binance"),
    ("BTCUSDT", "1h", "  "),
])
def test_validate_text_metadata_empty_refused(asset: str, timeframe: str, source: str) -> None:
    with pytest.raises(InvalidMetadataError):
        validate_text_metadata(asset, timeframe, source)


# --- Test 4 : CSV valide archivé ---

def test_archive_valid_csv(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    record = _archive(tmp_path, csv_path)
    assert record["event_type"] == "prospective_batch_archived"


# --- Test 5 : copie byte-for-byte identique ---

def test_archive_copies_bytes_exactly(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    source_bytes = csv_path.read_bytes()
    record = _archive(tmp_path, csv_path)
    stored_path = Path(record["stored_path"])
    assert stored_path.read_bytes() == source_bytes


# --- Test 6 : fichier source inchangé ---

def test_archive_does_not_modify_source(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    before = csv_path.read_bytes()
    _archive(tmp_path, csv_path)
    assert csv_path.read_bytes() == before


# --- Test 7 : nom du fichier archivé dérivé du SHA-256 ---

def test_archived_filename_derived_from_sha256(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    sha256 = compute_sha256(csv_path)
    reserve_dir = tmp_path / "reserve"
    record = _archive(tmp_path, csv_path)
    stored_path = Path(record["stored_path"])
    assert stored_path.name == f"{sha256}.csv"
    assert stored_path.parent == reserve_dir


# --- Test 8 : registre JSONL créé automatiquement ---

def test_registry_created_automatically(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    registry_path = tmp_path / "state" / "registry.jsonl"
    assert not registry_path.exists()
    archive_prospective_csv(
        path=csv_path,
        asset="BTCUSDT",
        timeframe="1h",
        source="binance-public",
        reserve_dir=tmp_path / "reserve",
        registry_path=registry_path,
        registered_at=FIXED_TS,
    )
    assert registry_path.exists()


# --- Test 9 : un seul enregistrement ajouté ---

def test_archive_adds_exactly_one_record(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    registry_path = tmp_path / "registry.jsonl"
    _archive(tmp_path, csv_path)
    records = read_jsonl(registry_path)
    assert len(records) == 1


# --- Test 10 : enregistrement conforme au schéma ---

def test_record_conforms_to_schema(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    record = _archive(tmp_path, csv_path)
    required = {
        "schema_version", "event_type", "batch_id", "data_role", "status",
        "asset", "timeframe", "source", "source_path", "stored_path",
        "file_sha256", "rows", "first_timestamp", "last_timestamp",
        "gaps_detected", "gaps", "registered_at",
        "performance_analysis_performed", "independence_claimed",
    }
    assert required.issubset(record.keys())


# --- Test 11 : data_role == prospective_holdout_candidate ---

def test_record_data_role(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    record = _archive(tmp_path, csv_path)
    assert record["data_role"] == "prospective_holdout_candidate"


# --- Test 12 : status == accumulating ---

def test_record_status_accumulating(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    record = _archive(tmp_path, csv_path)
    assert record["status"] == "accumulating"


# --- Test 13 : performance_analysis_performed is False ---

def test_record_performance_analysis_performed_false(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    record = _archive(tmp_path, csv_path)
    assert record["performance_analysis_performed"] is False


# --- Test 14 : independence_claimed is False ---

def test_record_independence_claimed_false(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    record = _archive(tmp_path, csv_path)
    assert record["independence_claimed"] is False


# --- Test 15 : timestamp injecté conservé exactement ---

def test_injected_registered_at_preserved(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    record = _archive(tmp_path, csv_path)
    assert record["registered_at"] == FIXED_TS


# --- Test 16 : timestamps OHLCV correctement sérialisés ---

def test_ohlcv_timestamps_serialized(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    record = _archive(tmp_path, csv_path)
    assert isinstance(record["first_timestamp"], str)
    assert isinstance(record["last_timestamp"], str)
    assert record["first_timestamp"] < record["last_timestamp"]


# --- Test 17 : gaps correctement décrits ---

def test_gaps_described_correctly(tmp_path: Path) -> None:
    csv_path = _write_csv_with_gap(tmp_path / "ohlcv_gap.csv")
    with pytest.warns(RuntimeWarning, match="gap"):
        record = archive_prospective_csv(
            path=csv_path,
            asset="BTCUSDT",
            timeframe="1h",
            source="binance-public",
            reserve_dir=tmp_path / "reserve",
            registry_path=tmp_path / "registry.jsonl",
            registered_at=FIXED_TS,
        )
    assert isinstance(record["gaps"], list)
    assert len(record["gaps"]) > 0
    assert record["gaps_detected"] == len(record["gaps"])
    gap = record["gaps"][0]
    assert "from" in gap
    assert "to" in gap
    assert "delta_seconds" in gap


# --- Test 18 : doublon refusé ---

def test_duplicate_refused(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    _archive(tmp_path, csv_path)
    with pytest.raises(DuplicateBatchError):
        _archive(tmp_path, csv_path)


# --- Test 19 : registre inchangé après doublon ---

def test_registry_unchanged_after_duplicate(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    registry_path = tmp_path / "registry.jsonl"
    _archive(tmp_path, csv_path)
    records_before = read_jsonl(registry_path)
    with pytest.raises(DuplicateBatchError):
        _archive(tmp_path, csv_path)
    records_after = read_jsonl(registry_path)
    assert records_before == records_after


# --- Test 20 : fichier archivé non écrasé ---

def test_reserve_file_not_overwritten(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    sha256 = compute_sha256(csv_path)
    reserve_dir = tmp_path / "reserve"
    reserve_dir.mkdir()
    dest_path = reserve_dir / f"{sha256}.csv"
    dest_path.write_bytes(b"preexisting")

    with pytest.raises(ReserveFileExistsError):
        archive_prospective_csv(
            path=csv_path,
            asset="BTCUSDT",
            timeframe="1h",
            source="binance-public",
            reserve_dir=reserve_dir,
            registry_path=tmp_path / "registry.jsonl",
            registered_at=FIXED_TS,
        )

    assert dest_path.read_bytes() == b"preexisting"


# --- Test 21 : CSV structurellement invalide refusé ---

def test_invalid_csv_refused(tmp_path: Path) -> None:
    csv_path = _write_invalid_csv(tmp_path / "invalid.csv")
    with pytest.raises(InvalidOHLCVError):
        _archive(tmp_path, csv_path)


# --- Test 22a : aucune archive ou entrée créée après CSV invalide ---

def test_no_files_created_after_invalid_csv(tmp_path: Path) -> None:
    csv_path = _write_invalid_csv(tmp_path / "invalid.csv")
    reserve_dir = tmp_path / "reserve"
    registry_path = tmp_path / "registry.jsonl"

    with pytest.raises(InvalidOHLCVError):
        archive_prospective_csv(
            path=csv_path,
            asset="BTCUSDT",
            timeframe="1h",
            source="binance-public",
            reserve_dir=reserve_dir,
            registry_path=registry_path,
            registered_at=FIXED_TS,
        )

    assert not registry_path.exists()
    assert not reserve_dir.exists() or not any(reserve_dir.iterdir())


# --- Test 22b : aucune archive ou entrée créée après métadonnées invalides ---

def test_no_files_created_after_invalid_metadata(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    reserve_dir = tmp_path / "reserve"
    registry_path = tmp_path / "registry.jsonl"

    with pytest.raises(InvalidMetadataError):
        archive_prospective_csv(
            path=csv_path,
            asset="",
            timeframe="1h",
            source="binance-public",
            reserve_dir=reserve_dir,
            registry_path=registry_path,
            registered_at=FIXED_TS,
        )

    assert not registry_path.exists()
    assert not reserve_dir.exists() or not any(reserve_dir.iterdir())


# --- Test 23 : aucun champ de performance interdit ---

def test_no_forbidden_performance_fields(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    record = _archive(tmp_path, csv_path)
    forbidden = FORBIDDEN_PERFORMANCE_FIELDS.intersection(record.keys())
    assert forbidden == set(), f"Forbidden fields found: {forbidden}"


# --- Test 24 : aucun changement dans state/trades.jsonl ou state/hypotheses.jsonl ---

def test_no_change_to_state_trades_or_hypotheses(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "ohlcv.csv")
    trades_path = tmp_path / "state" / "trades.jsonl"
    hypotheses_path = tmp_path / "state" / "hypotheses.jsonl"

    archive_prospective_csv(
        path=csv_path,
        asset="BTCUSDT",
        timeframe="1h",
        source="binance-public",
        reserve_dir=tmp_path / "reserve",
        registry_path=tmp_path / "state" / "prospective_reserve.jsonl",
        registered_at=FIXED_TS,
    )

    assert not trades_path.exists()
    assert not hypotheses_path.exists()
