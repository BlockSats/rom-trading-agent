from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def generate_sample_ohlcv(rows: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h")

    trend = np.linspace(100.0, 120.0, rows)
    cycle = np.sin(np.linspace(0, 6 * np.pi, rows)) * 18.0
    noise = rng.normal(0.0, 1.0, rows)
    close = trend + cycle + noise

    open_prices = np.empty(rows, dtype="float64")
    open_prices[0] = close[0]
    if rows > 1:
        open_prices[1:] = close[:-1] + rng.normal(0.0, 0.4, rows - 1)

    spread_high = np.abs(rng.normal(0.6, 0.2, rows))
    spread_low = np.abs(rng.normal(0.6, 0.2, rows))
    high = np.maximum(open_prices, close) + spread_high
    low = np.minimum(open_prices, close) - spread_low
    volume = rng.integers(100, 1000, rows)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing required OHLCV columns: {', '.join(missing)}")


def _to_timedelta(value: Any) -> pd.Timedelta:
    if isinstance(value, pd.Timedelta):
        return value
    if hasattr(value, "total_seconds"):
        return pd.to_timedelta(value)
    if isinstance(value, (int, float)):
        return pd.to_timedelta(float(value), unit="s")
    return pd.to_timedelta(value)


def validate_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    validated = df.copy()
    _ensure_required_columns(validated)

    if not pd.api.types.is_datetime64_any_dtype(validated["timestamp"]):
        raise ValueError("timestamp column must be datetime")

    if validated["timestamp"].isna().any():
        raise ValueError("timestamp column contains missing values")

    if validated["timestamp"].duplicated().any():
        raise ValueError("duplicate timestamps are not allowed")

    if not validated["timestamp"].is_monotonic_increasing:
        raise ValueError("rows must be sorted by timestamp ascending")

    for column in ["open", "high", "low", "close", "volume"]:
        if not pd.api.types.is_numeric_dtype(validated[column]):
            raise ValueError(f"{column} column must be numeric")
        if validated[column].isna().any():
            raise ValueError(f"{column} column contains missing values")

    if (validated["volume"] < 0).any():
        raise ValueError("volume must be greater than or equal to 0")

    too_low = validated["high"] < validated[["open", "close"]].max(axis=1)
    if too_low.any():
        raise ValueError("high must be greater than or equal to max(open, close)")

    too_high = validated["low"] > validated[["open", "close"]].min(axis=1)
    if too_high.any():
        raise ValueError("low must be less than or equal to min(open, close)")

    return validated[REQUIRED_OHLCV_COLUMNS].reset_index(drop=True)


def detect_time_gaps(df: pd.DataFrame, expected_frequency: Any | None = None) -> list[dict[str, Any]]:
    if df.empty or len(df) < 2:
        return []

    ordered = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.to_datetime(ordered["timestamp"])
    deltas = timestamps.diff().dropna()
    if deltas.empty:
        return []

    if expected_frequency is None:
        delta_seconds = [int(delta.total_seconds()) for delta in deltas if delta.total_seconds() > 0]
        if not delta_seconds:
            return []
        expected_seconds = Counter(delta_seconds).most_common(1)[0][0]
        expected_delta = pd.to_timedelta(expected_seconds, unit="s")
    else:
        expected_delta = _to_timedelta(expected_frequency)

    gaps: list[dict[str, Any]] = []
    for index, delta in enumerate(deltas, start=1):
        if delta > expected_delta:
            previous_timestamp = timestamps.iloc[index - 1]
            current_timestamp = timestamps.iloc[index]
            gaps.append(
                {
                    "from": previous_timestamp.isoformat(),
                    "to": current_timestamp.isoformat(),
                    "delta_seconds": int(delta.total_seconds()),
                }
            )
    return gaps


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    df = pd.read_csv(file_path)
    _ensure_required_columns(df)

    loaded = df.copy()
    loaded["timestamp"] = pd.to_datetime(loaded["timestamp"], errors="raise")
    if loaded["timestamp"].isna().any():
        raise ValueError("timestamp column contains missing values")

    for column in ["open", "high", "low", "close", "volume"]:
        loaded[column] = pd.to_numeric(loaded[column], errors="raise")
        if loaded[column].isna().any():
            raise ValueError(f"{column} column contains missing values")

    loaded = loaded.sort_values("timestamp").reset_index(drop=True)
    validated = validate_ohlcv_dataframe(loaded)

    gaps = detect_time_gaps(validated)
    if gaps:
        warnings.warn(f"Detected {len(gaps)} gap(s) in OHLCV data", RuntimeWarning, stacklevel=2)

    return validated
