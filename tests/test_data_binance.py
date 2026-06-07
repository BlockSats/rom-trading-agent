from __future__ import annotations

from pathlib import Path

from trading_agent.market_data.binance import write_ohlcv_csv
from trading_agent.data import load_ohlcv_csv


def test_write_ohlcv_csv_creates_compatible_file(tmp_path: Path) -> None:
    rows = [
        {
            "timestamp": 1_704_067_200_000,
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 123.45,
        },
        {
            "timestamp": 1_704_070_800_000,
            "open": 105.0,
            "high": 115.0,
            "low": 95.0,
            "close": 108.0,
            "volume": 234.56,
        },
    ]

    output = tmp_path / "BTCUSDT_1h.csv"
    write_ohlcv_csv(rows, output)

    loaded = load_ohlcv_csv(output)

    assert len(loaded) == 2
    assert list(loaded.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert loaded["close"].iloc[-1] == 108.0
