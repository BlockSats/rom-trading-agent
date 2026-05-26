# rom-trading-agent

Local-first paper-trading research agent foundation for Linux systems.

## Safety

Phase 1 is paper trading only.
No live trading, no exchange order execution, no private exchange keys, no cloud deployment, and no LLM API calls are implemented here.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Basic Commands

```bash
trading-agent check
trading-agent run-once
trading-agent score
trading-agent reflect
trading-agent backtest-csv data/btc_usdt_1h.csv
pytest
```

## Phase 2 CSV Backtest

Phase 2 adds a read-only historical backtest path for local CSV OHLCV files.

Expected CSV columns:

```text
timestamp,open,high,low,close,volume
```

Rules:

- `timestamp` is parsed as datetime.
- `open`, `high`, `low`, `close`, and `volume` must be numeric.
- Rows are sorted by timestamp ascending.
- Duplicate timestamps are rejected.
- Missing required columns raise `ValueError`.
- Invalid OHLC values raise `ValueError`.
- Gaps are reported as warnings, not hard failures.

Example:

```bash
trading-agent backtest-csv data/btc_usdt_1h.csv
```

The CSV backtest writes `outputs/backtest_report.json` and `outputs/backtest_trades.jsonl` only.
It does not modify `state/trades.jsonl`, strategy files, or strategy history.

## Repository Layout

```text
config/        YAML configuration
data/          Local sample data and placeholders
state/         Append-only state files and history
src/trading_agent/
tests/         Phase 1 tests
```

## Phase 1 Scope

- Python project scaffold
- Safe config loading and validation
- Deterministic sample data
- RSI indicator
- Simple paper broker
- One-pass strategy runner
- Local CSV OHLCV loading
- Deterministic historical backtesting
- Append-only storage helpers
- Deterministic scoring
- Deterministic fallback reflection
- Typer CLI commands

## Not Implemented Yet

- Live trading
- Real exchange integration
- Private API keys
- Autonomous infinite loops
- Docker
- Railway
- Hermes
- Cloud deployment
- LLM API calls
