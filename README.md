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
pytest
```

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

