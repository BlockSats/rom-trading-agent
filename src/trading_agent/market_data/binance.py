from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def fetch_binance_ohlcv(
    symbol: str,
    interval: str,
    limit: int = 500,
) -> list[dict[str, object]]:
    """Fetch public OHLCV candles from Binance spot REST API.

    No API key is required for this endpoint.
    """
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000")

    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }

    url = f"{BINANCE_KLINES_URL}?{urlencode(params)}"

    with urlopen(url, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if isinstance(payload, dict) and "code" in payload:
        raise ValueError(f"Binance API error: {payload}")

    rows: list[dict[str, object]] = []

    for candle in payload:
        rows.append(
            {
                # Binance returns timestamps in milliseconds.
                "timestamp": int(candle[0]),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
            }
        )

    return rows


def write_ohlcv_csv(rows: list[dict[str, object]], output_path: str | Path) -> Path:
    """Write OHLCV rows to a CSV compatible with the existing loader."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OHLCV_COLUMNS)
        writer.writeheader()

        for row in rows:
            # Convert milliseconds to ISO-like UTC string.
            timestamp_ms = int(row["timestamp"])
            timestamp_seconds = timestamp_ms / 1000

            from datetime import datetime, timezone

            timestamp = datetime.fromtimestamp(timestamp_seconds, tz=timezone.utc)
            output_row = {
                "timestamp": timestamp.isoformat(),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
            writer.writerow(output_row)

    return path
