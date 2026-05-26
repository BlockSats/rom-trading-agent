from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trading_agent.data import detect_time_gaps, load_ohlcv_csv


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    df.to_csv(path, index=False)
    return path


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 03:00:00"]
            ),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.5, 103.0],
            "low": [99.0, 100.5, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 20.0, 30.0],
        }
    )


def test_valid_csv_loads(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "ohlcv.csv", _valid_frame())
    loaded = load_ohlcv_csv(csv_path)
    assert list(loaded.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert pd.api.types.is_datetime64_any_dtype(loaded["timestamp"])


def test_missing_columns_raise_value_error(tmp_path: Path) -> None:
    frame = _valid_frame().drop(columns=["volume"])
    csv_path = _write_csv(tmp_path / "missing.csv", frame)
    with pytest.raises(ValueError):
        load_ohlcv_csv(csv_path)


def test_duplicate_timestamps_raise_value_error(tmp_path: Path) -> None:
    frame = _valid_frame()
    frame.loc[2, "timestamp"] = frame.loc[1, "timestamp"]
    csv_path = _write_csv(tmp_path / "dup.csv", frame)
    with pytest.raises(ValueError):
        load_ohlcv_csv(csv_path)


def test_invalid_ohlc_values_raise_value_error(tmp_path: Path) -> None:
    frame = _valid_frame()
    frame.loc[1, "high"] = 100.0
    csv_path = _write_csv(tmp_path / "bad_ohlc.csv", frame)
    with pytest.raises(ValueError):
        load_ohlcv_csv(csv_path)


def test_negative_volume_raises_value_error(tmp_path: Path) -> None:
    frame = _valid_frame()
    frame.loc[1, "volume"] = -1.0
    csv_path = _write_csv(tmp_path / "bad_volume.csv", frame)
    with pytest.raises(ValueError):
        load_ohlcv_csv(csv_path)


def test_rows_are_sorted_ascending(tmp_path: Path) -> None:
    frame = _valid_frame().iloc[[2, 0, 1]].reset_index(drop=True)
    csv_path = _write_csv(tmp_path / "unsorted.csv", frame)
    loaded = load_ohlcv_csv(csv_path)
    assert loaded["timestamp"].is_monotonic_increasing is True


def test_time_gaps_are_detected(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "gaps.csv", _valid_frame())
    loaded = load_ohlcv_csv(csv_path)
    gaps = detect_time_gaps(loaded)
    assert len(gaps) == 1
    assert gaps[0]["from"] == "2024-01-01T01:00:00"
    assert gaps[0]["to"] == "2024-01-01T03:00:00"
    assert gaps[0]["delta_seconds"] == 7200
