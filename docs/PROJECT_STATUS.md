# Project Status

## Current Version

Current documentation release: `v0.40-walk-forward-design`
Functional baseline: `v0.39-assets-report-inventory`

## Test Baseline

```
240 passed, 4 warnings
```

The 4 warnings come from OHLCV gap detection tests in `tests/test_data_csv.py` and `tests/test_cli_backtest.py`. They are expected.

## Version History

| Tag | Description |
|-----|-------------|
| v0.40 | Walk-Forward Analysis design document |
| v0.39 | Assets report inventory (read-only) |
| v0.38 | Project status documentation update |
| v0.37 | Assets report display filters (`--asset`, `--strategy`) |
| v0.36 | Assets report window robustness diagnostics |
| v0.35 | Show assets comparison report CLI command |
| v0.34 | Multi-asset strategy comparison |
| v0.33 | Window robustness diagnostics |
| v0.32 | Classification reasons in backtest report |
| v0.31 | Donchian breakout candidate strategy |
| v0.30 | Strategy ID reflection |
| v0.29 | ATR risk model |

## Strategy Candidates

| ID | Status |
|----|--------|
| `rsi_baseline` | active (reference) |
| `ema_atr_trend` | candidate |
| `donchian_breakout` | candidate |

Candidates are deterministic and backtestable. None are declared profitable.

## Known Limits

- No live trading
- No real exchange orders
- No Walk-Forward Analysis implemented yet — see `docs/WALK_FORWARD_ANALYSIS_DESIGN.md`
- No DSR / PBO (Deflated Sharpe Ratio / Probability of Backtest Overfitting)
- No automatic strategy selection
- No promotion from candidate to production
