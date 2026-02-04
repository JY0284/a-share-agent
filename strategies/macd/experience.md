---
name: macd_strategy
description: MACD-based strategy: signal line cross, histogram, or divergence for entries and exits.
tags: [strategy, MACD, 趋势, 动量, 策引]
---

## Core rule
MACD = fast EMA − slow EMA; **Signal line** = EMA of MACD (e.g. 9). Common signals: MACD crossing above/below signal (crossover), or histogram sign change. Use with **chronological** data and avoid lookahead (e.g. signal on close of *t*, trade from *t+1*).

## Components (from doc)
- **MACD line**: Difference between two EMAs (e.g. 12 and 26).
- **Signal line**: EMA of MACD (e.g. 9).
- **Histogram**: MACD − Signal (optional; zero cross = crossover).

## Typical signals
- **Long**: MACD crosses **above** signal line (or histogram turns positive).
- **Short / exit**: MACD crosses **below** signal line (or histogram turns negative).
- **Divergence** (advanced): Price makes new high/low but MACD does not; can warn of reversal.

## Parameters
- Fast EMA: 12, Slow EMA: 26, Signal EMA: 9 (common defaults).
- Different periods change sensitivity (shorter = more signals, more noise).

## Use case
Trend and momentum confirmation; often combined with other filters (e.g. trend filter, volume) to reduce false signals.

## See also
- `dual_moving_average`: another trend-following code strategy
- `basic_concepts`: technical indicators, signal states
- 策引 doc: [MACD策略](https://docs.myinvestpilot.com/docs/strategies/macd-strategy)
