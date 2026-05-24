from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json

from trading_agent.config import load_strategy
from trading_agent.data import generate_sample_ohlcv
from trading_agent.storage import append_jsonl, ensure_state_files
from trading_agent.strategy import run_strategy_on_dataframe


def run_once() -> dict[str, Any]:
    ensure_state_files()
    strategy = load_strategy()
    data = generate_sample_ohlcv()
    closed_trades = run_strategy_on_dataframe(data, strategy)

    trade_path = Path("state/trades.jsonl")
    for trade in closed_trades:
        append_jsonl(trade_path, trade)

    heartbeat = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "closed_trades": len(closed_trades),
    }
    Path("state/heartbeat.json").write_text(json.dumps(heartbeat, sort_keys=True), encoding="utf-8")

    total_net_pnl = sum(float(trade.get("net_pnl", 0.0)) for trade in closed_trades)
    return {
        "closed_trades": len(closed_trades),
        "total_net_pnl": float(total_net_pnl),
        "heartbeat": heartbeat,
    }

