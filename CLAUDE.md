# CLAUDE.md — rom-trading-agent

## Project context

`rom-trading-agent` is a Python trading research agent.

It is a local-first research, backtesting, scoring, and paper-trading project.

It is not a live-money trading bot.

The main CLI entrypoint is `tradebot`.

The project currently includes:

- CSV OHLCV loading and validation
- Binance public market data utilities
- Strategy signal generation
- Backtesting
- Trade scoring
- Strategy comparison
- Research policy configuration
- Paper-trading simulation
- CLI and unit tests

## Safety rules

Never enable live trading by default.

Never place real orders.

Never add real exchange API keys.

Never create, modify, or commit `.env` files containing secrets.

Never print secrets in terminal output.

Never assume Binance, Kraken, or any exchange integration is allowed to trade real funds.

All exchange integrations must default to public data, paper mode, or simulation mode.

Do not add automatic strategy promotion.

Do not add `best_strategy`.

Do not add automatic strategy selection unless the user explicitly requests it.

Any new strategy must be deterministic, testable, and backtestable.

## Development workflow

Before changing code:

1. Check Git state with `git status --short -uall`.
2. Check the current branch with `git branch --show-current`.
3. Inspect the relevant files in `src/trading_agent/`.
4. Inspect the matching tests in `tests/`.
5. Keep the change small and focused.

After changing code:

1. Run the full test suite with `python -m pytest`.
2. If CLI behavior changes, add or update CLI tests.
3. If config behavior changes, add or update config tests.
4. Show the Git state with `git status --short -uall`.
5. Show the diff with `git diff` before proposing a commit.

## Git rules

Work on a feature branch.

Do not push directly to `main`.

Do not rewrite shared history.

Do not run `git reset --hard` unless explicitly requested.

Do not run `git clean -fd` unless explicitly requested.

Do not commit, tag, merge, or push without explicit user approval.

Commit messages should be short and explicit.

## Code style

Keep code readable for a beginner-to-intermediate Python user.

Prefer explicit names over clever abstractions.

Avoid unnecessary abstractions.

Avoid large rewrites.

Add comments only when they clarify non-obvious logic.

Use deterministic behavior in tests.

Avoid network-dependent tests unless mocked or safely isolated.

## Useful commands

Useful local commands:

- `source .venv/bin/activate`
- `python -m pytest`
- `tradebot --help`
- `tradebot status`
- `tradebot config`
- `tradebot config --section strategy`
- `tradebot config --section goal`

## Current development direction

Proceed in small, testable phases.

The next preferred phase is:

`v0.25` — add a read-only CLI command to display `config/research_policy.yaml` in a human-readable format.

Constraints for `v0.25`:

- read `config/research_policy.yaml`
- display the research policy clearly
- do not modify the YAML file
- do not modify `strategy.yaml`
- do not recalculate backtests
- do not add `best_strategy`
- do not add automatic strategy selection
- do not add live trading
- do not touch `.env`
