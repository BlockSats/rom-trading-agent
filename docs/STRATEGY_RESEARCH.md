# Strategy Research

## Role of the Project

`rom-trading-agent` is a local research lab for trading strategy candidates. Its scope is:

- loading and validating OHLCV data;
- running deterministic backtests;
- scoring closed trades;
- comparing research reports;
- simulating paper trading.

The project is not a live trading bot. Live trading is not enabled by default and no strategy candidate should be treated as ready for real orders without a separate safety gate, dedicated approval, and additional validation.

## Strategy Candidates

A strategy candidate is a deterministic, testable idea that can be expressed with explicit entry, exit, risk, and cost assumptions. A candidate is never assumed to be profitable before validation. It must be compared against the same data, fees, risk rules, and metrics as other candidates.

The current RSI baseline is a documented candidate. Future candidates should be added only when they can be tested without changing the safety assumptions of the project.

## Priority Markets

Initial research should focus on crypto CEX market data because the existing project already uses public crypto OHLCV data.

Priority assets:

- BTC;
- ETH;
- SOL.

Later extensions can include:

- BNB;
- XRP;
- DOGE;
- ADA;
- AVAX;
- LINK;
- NEAR.

An aggressive basket can be considered later only when the input data is clean, validated, and comparable across assets.

## Priority Timeframes

Initial research should prioritize:

- 1h for primary strategy development;
- 4h for slower confirmation and regime checks;
- daily for validation context.

The project should avoid 5m and 15m strategies at the beginning because they are more sensitive to fees, slippage, noisy entries, exchange outages, and data quality issues.

## Candidate Families

Strategy families to document and test over time:

- RSI baseline, already present;
- EMA / ATR trend following;
- breakout / Donchian;
- mean reversion as a secondary hypothesis;
- ICT-inspired ideas later, only as explicit and testable patterns.

No family should be promoted beyond candidate status without evidence from repeatable backtests and robustness checks.

## Comparison Rules

Candidates should be compared using:

- the same CSV data;
- the same fee assumptions when fees are part of the test;
- the same risk rules;
- the same scoring metrics;
- the same reporting workflow.

Do not compare candidates using one trade, one isolated window, or one favorable market period. A comparison should include enough trades and enough market variety to make the result useful.

## Anti-Overfitting Rules

Avoid excessive parameter optimization. A candidate that works only after many parameter tweaks is weaker than a simpler candidate that remains stable across regimes.

Initial rejection or caution should apply when:

- conclusions rely on fewer than 100 closed trades;
- performance appears only in one market regime;
- results depend heavily on one parameter value;
- a small parameter change reverses the result;
- a strategy works only on one asset without a clear reason.

Walk-forward validation should be added later before any candidate is considered for broader paper trading use.
