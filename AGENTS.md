# AGENTS.md — rom-trading-agent

## Project context

This repository contains `rom-trading-agent`, a Python trading research agent.

It is a research, backtesting, scoring, and paper-trading project. It is not a live-money trading bot.

The current CLI entrypoint is:

```bash
tradebot
```

The project includes:

* CSV OHLCV data loading and validation
* Binance public market data utilities
* RSI indicator logic
* Strategy scoring
* Backtesting
* Paper broker simulation
* Local state storage
* Research cycle commands
* Controlled research reflection
* Robustness validation
* CLI tests and Python unit tests

## Core safety rules

* Never enable live trading by default.
* Never place real orders.
* Never add real exchange API keys.
* Never create, modify, or commit `.env` files containing secrets.
* Never print secrets in terminal output.
* Never assume Binance, Kraken, or any exchange integration is allowed to trade real funds.
* All exchange integrations must default to public data, paper mode, or simulation mode.
* Any future live-trading feature must require an explicit safety gate and dedicated user approval.
* Any new strategy must be deterministic, testable, and backtestable.

## Development workflow

Before changing code:

1. Inspect the current architecture.
2. Read the relevant files in `src/trading_agent/`.
3. Read the matching tests in `tests/`.
4. Keep changes small and focused.
5. Prefer incremental improvements over large rewrites.

After changing code:

1. Run the test suite:

```bash
python -m pytest
```

2. If CLI behavior changes, add or update CLI tests.
3. If config behavior changes, add or update config tests.
4. If market data behavior changes, add or update market data tests.
5. If research cycle behavior changes, add or update research cycle tests.
6. Show the diff before proposing a commit.

## Code style

* Keep the code readable for a beginner-to-intermediate Python user.
* Prefer explicit names over clever abstractions.
* Avoid unnecessary abstractions.
* Avoid large rewrites unless specifically requested.
* Add comments only when they clarify non-obvious logic.
* Keep functions small when practical.
* Use deterministic behavior in tests.
* Avoid network-dependent tests unless they are mocked or safely isolated.

## Project commands

Create or activate the environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project locally:

```bash
pip install -e .
```

Run tests:

```bash
python -m pytest
```

Collect tests only:

```bash
python -m pytest --collect-only -q
```

Useful CLI checks:

```bash
tradebot --help
tradebot status
tradebot config
tradebot config --section strategy
tradebot config --section goal
```

## Git rules

* Work on a feature branch.
* Do not push directly to `main`.
* Do not rewrite history on shared branches.
* Do not use `git reset --hard` unless explicitly requested.
* Do not use `git clean -fd` unless explicitly requested.
* Show `git status --short -uall` before committing.
* Show `git diff` before committing.
* Commit messages should be short and explicit.

Recommended branch naming:

```bash
feature/config-cli-output
feature/research-cycle-improvement
fix/binance-data-validation
```

## Current quality baseline

The expected baseline is:

```bash
python -m pytest
```

Expected result:

```text
63 passed
```

Warnings related to intentional OHLCV gap detection are acceptable unless the task specifically concerns warning handling.

## Specific guidance for CLI changes

When modifying `src/trading_agent/cli.py`:

* Keep existing commands backward compatible.
* Do not remove existing command options without explicit approval.
* Update or add tests under `tests/test_cli_*.py`.
* Ensure commands return useful output.
* Ensure invalid inputs produce clear errors.
* Run the full test suite after changes.

## Specific guidance for config behavior

When modifying config-related behavior:

* Preserve existing config schema unless explicitly asked.
* Validate user input clearly.
* Keep output human-readable.
* If `tradebot config` is called without a section, it should print a useful readable summary.
* If `tradebot config --section strategy` is called, it should print only the strategy section.
* If `tradebot config --section goal` is called, it should print only the goal section.
* Invalid sections should fail clearly and be covered by tests.

## Specific guidance for market data

When modifying market data code:

* Public data access is allowed.
* Do not require API keys for public market data.
* Do not add real trading endpoints.
* Do not place orders.
* Prefer pure parsing and validation tests.
* Avoid fragile live-network tests.

## Specific guidance for research and robustness

When modifying research cycle, reflection, or robustness logic:

* Keep behavior deterministic.
* Do not allow uncontrolled self-modification.
* Proposed strategy changes must stay within safe bounds.
* Robustness validation should penalize unstable or overfit behavior.
* Add tests for edge cases and failure paths.

## First Codex task

The first development task is:

Improve `tradebot config` so that it prints a readable default output when no section is provided.

Constraints:

* Keep the change small.
* Do not break existing tests.
* Add or update CLI tests if needed.
* Run `python -m pytest`.
* Show the diff before proposing a commit.


