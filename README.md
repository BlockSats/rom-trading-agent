# rom-trading-agent

Local-first Python trading research agent for backtesting, scoring, and paper trading.

Not a live trading bot. Research and simulation only.

See [Project Status](docs/PROJECT_STATUS.md) for the current version and test baseline.

## Safety

- No live trading
- No real exchange orders
- No API keys required for local workflows
- No secrets in the repository
- Paper-only, research-first

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Current Capabilities

- CSV OHLCV loading and validation
- Local backtesting
- Multi-window strategy comparison
- Multi-asset strategy comparison
- Report display
- Window robustness diagnostics
- Display filters by asset and strategy
- Strategy candidates: `rsi_baseline`, `ema_atr_trend`, `donchian_breakout`

## Useful Commands

```bash
python -m pytest

tradebot check
tradebot backtest-csv data/btc_usdt_1h.csv

tradebot compare-strategies-windows-csv data/btc_usdt_1h.csv
tradebot compare-strategies-assets-csv \
  --asset BTCUSDT:data/btc_usdt_1h.csv \
  --asset ETHUSDT:data/eth_usdt_1h.csv

tradebot show-comparison-report
tradebot show-assets-comparison-report
tradebot show-assets-comparison-report --asset BTCUSDT
tradebot show-assets-comparison-report --strategy rsi_baseline
```

## Repository Layout

```text
config/        YAML configuration
data/          Local sample data and placeholders
docs/          Project documentation
state/         Append-only state files and history
src/trading_agent/
tests/
outputs/       Generated reports (not committed by default)
```

## Limits

- No live trading
- No strategy declared profitable
- Candidate strategies only — not validated for production use
- No Walk-Forward Analysis yet
- No DSR / PBO yet
- No auto-selection of strategies
