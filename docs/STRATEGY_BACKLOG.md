# Strategy Backlog

Statuses:

- documented;
- planned;
- blocked;
- testing;
- rejected;
- accepted_for_paper.

## Candidate 001 - RSI Baseline

- Status: documented
- Idea: Use the existing RSI entry and RSI take-profit logic as the baseline strategy candidate.
- Data needed: Clean OHLCV for BTC, ETH, and SOL.
- Timeframes: 1h first, then 4h and daily validation.
- Metrics to watch: Expectancy, profit factor, drawdown, closed trades, winrate.
- Rejection reason: Negative expectancy, profit factor below 1, unstable results across assets or regimes.

## Candidate 002 - EMA / ATR Trend Following

- Status: planned
- Idea: Follow trend direction with EMA structure and use ATR for volatility-aware exits or stops.
- Data needed: Clean OHLCV with enough history for trend and volatility windows.
- Timeframes: 1h and 4h.
- Metrics to watch: Expectancy, profit factor, drawdown, trade count.
- Rejection reason: Whipsaw losses, excessive drawdown, or strong sensitivity to EMA/ATR parameters.

## Candidate 003 - Donchian Breakout

- Status: planned
- Idea: Test breakout entries when price clears a recent high or low range.
- Data needed: Clean OHLCV with sustained trend and range periods.
- Timeframes: 1h, 4h, daily validation.
- Metrics to watch: Profit factor, expectancy, drawdown, number of false breakouts.
- Rejection reason: Too many failed breakouts, poor performance after fees, or returns concentrated in one period.

## Candidate 004 - RSI Mean Reversion

- Status: planned
- Idea: Test whether RSI extremes can identify temporary overextension and reversal zones.
- Data needed: Clean OHLCV across ranging and trending regimes.
- Timeframes: 1h first, 4h validation.
- Metrics to watch: Expectancy, profit factor, drawdown, average win/loss.
- Rejection reason: Large losses during trends or negative expectancy after costs.

## Candidate 005 - Volatility Filter

- Status: planned
- Idea: Add a volatility condition to allow or block entries when market movement is too low or too high.
- Data needed: Clean OHLCV with enough history for volatility estimates.
- Timeframes: 1h and 4h.
- Metrics to watch: Trade count, expectancy, drawdown, filtered trade quality.
- Rejection reason: Too few trades, no improvement over baseline, or overfitted thresholds.

## Candidate 006 - Multi-Timeframe Daily + 4h Filter

- Status: planned
- Idea: Use daily context and 4h confirmation to filter lower timeframe entries.
- Data needed: Aligned OHLCV data for daily, 4h, and 1h where needed.
- Timeframes: Daily and 4h, with optional 1h execution later.
- Metrics to watch: Expectancy, drawdown, trade count, regime stability.
- Rejection reason: Inconsistent alignment, too few trades, or no improvement over single-timeframe tests.

## Candidate 007 - Funding / OI Filter, Later

- Status: planned
- Idea: Use funding or open interest context as a filter once clean derivatives data is available.
- Data needed: OHLCV plus funding and open interest history from reliable sources.
- Timeframes: 1h, 4h, daily validation.
- Metrics to watch: Expectancy, drawdown, trade count, filter impact.
- Rejection reason: Data quality issues, unavailable history, or no measurable improvement over OHLCV-only candidates.

## Candidate 008 - ICT-Inspired Liquidity Sweep, Later

- Status: planned
- Idea: Convert liquidity sweep ideas into explicit, testable price patterns.
- Data needed: Clean OHLCV with enough granularity to define sweeps without discretionary interpretation.
- Timeframes: 1h and 4h first; lower timeframes only after data quality review.
- Metrics to watch: Expectancy, profit factor, drawdown, false sweep rate.
- Rejection reason: Pattern cannot be defined deterministically, too few trades, or performance concentrated in one period.
