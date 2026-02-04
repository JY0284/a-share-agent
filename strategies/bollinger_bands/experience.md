---
name: bollinger_bands
description: Bollinger Bands strategy: mean reversion or breakout using upper/lower bands and width.
tags: [strategy, 布林带, Bollinger, volatility, 策引]
---

## Core rule
**Middle** = moving average (e.g. 20-period SMA). **Upper** = middle + k×std, **Lower** = middle − k×std (often k=2). Price near lower band → potential bounce (mean reversion); price breaking upper band → potential breakout (trend). Strategy must choose one style or combine with other conditions.

## Components (from doc)
- **Middle band**: SMA (often 20).
- **Upper / Lower**: Middle ± multiple of rolling standard deviation (often 2).
- **Bandwidth**: (Upper − Lower) / Middle; measures volatility.

## Common logic
- **Mean reversion**: Buy when price touches or goes below lower band; sell when price touches or goes above upper band (or take profit at middle).
- **Breakout**: Buy when price closes above upper band (and maybe with volume); exit when price returns inside or breaks below middle/lower.
- **Squeeze**: Low bandwidth → potential big move; direction from other signals.

## Parameters
- Period: 20 common; 50 for longer term.
- Std multiplier: 2 typical; larger = wider bands, fewer touches.

## Caveat
In **strong trends**, mean-reversion entries can fail (price can stay at lower band or break further). Often combined with trend filter (e.g. only mean-revert in range-bound regimes).

## See also
- `basic_concepts`: technical indicators, mean reversion vs trend
- `risk_metrics`: volatility for band width
- 策引 doc: [布林带策略](https://docs.myinvestpilot.com/docs/strategies/bollinger-bands)
