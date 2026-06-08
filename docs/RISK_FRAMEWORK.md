# Risk Framework

## Initial Risk Frame

The project should favor stable and repeatable research results over aggressive returns. Risk settings are research assumptions, not promises of performance.

Initial risk guidelines:

- target risk per trade: about 0.5%;
- initial rejection drawdown threshold: about 20%;
- paper daily loss limit: about 2%;
- one open position at a time at the beginning;
- regularity and controlled downside are more important than aggressive upside.

These values are starting assumptions for research. They can be revised only with explicit tests and clear documentation.

## Priority Metrics

Strategy candidates should be evaluated primarily with:

- expectancy;
- profit factor;
- drawdown;
- number of closed trades;
- winrate as a supporting metric only.

Winrate alone is not enough. A candidate with a high winrate can still be weak if losses are large, expectancy is negative, or drawdown is unstable.

## Rejection Criteria

A strategy candidate should be rejected or kept out of paper testing when one or more of these conditions apply:

- insufficient number of closed trades;
- profit factor below 1;
- negative expectancy;
- drawdown above the current rejection threshold;
- unstable behavior across market regimes;
- performance concentrated in one narrow period;
- results dependent on one asset without similar behavior elsewhere.

These criteria are meant to prevent weak candidates from advancing because of one favorable sample.

## Caution Criteria

Apply extra skepticism when a candidate shows:

- results that look too perfect;
- too few trades;
- stops that are too wide;
- strong sensitivity to small parameter changes;
- effectiveness on only one asset;
- strong performance only in one market regime;
- large gains from one or two outlier trades.

A candidate that triggers caution criteria may still be useful for research, but it should not be treated as accepted for paper trading without more validation.
