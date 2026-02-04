---
name: dual_moving_average
description: Classic trend-following strategy using short- and long-period MA crossover (golden cross / death cross).
tags: [strategy, 双均线, MA, 金叉, 死叉, trend, 策引]
---

## Core rule
- **Golden cross (金叉)**: Short MA crosses **above** long MA → **BUY**
- **Death cross (死叉)**: Short MA crosses **below** long MA → **SELL**
- Use **chronological** daily data; compute signals on close of day *t*, execute from day *t+1* to avoid lookahead.

## How it works
1. Compute two MAs: short (e.g. 10d) and long (e.g. 30d).
2. **Entry**: short MA moves from below to above long MA (golden cross).
3. **Exit**: short MA moves from above to below long MA (death cross).
4. MAs filter noise and reflect trend; no need to predict turning points.

## Parameter sets (from 策引 doc)
| Short / Long | Use case |
|--------------|----------|
| 5 / 20       | More signals, more false; short-term |
| 10 / 30      | Balanced; short-to-mid term |
| 20 / 60      | Fewer, higher-quality signals; mid-term |
| 50 / 200     | Long-term; famous "golden cross" |

## MA type
- **SMA**: Equal weight, stable, most common.
- **EMA**: More weight on recent price, more responsive.

## Market behavior (from doc)
- **Bull**: Works well; can shorten periods and increase exposure.
- **Bear**: Exits in time; use longer periods and smaller size.
- **Sideways**: Many false signals; reduce trading or add filters (e.g. volume, RSI, MACD).

## Improvements (from doc)
1. **Volume confirmation**: Require higher volume on golden cross.
2. **Three MAs**: Add mid-term MA; trade only when all three align.
3. **Other indicators**: RSI overbought/oversold, MACD, support/resistance.

## Common pitfalls
- Computing MAs on unsorted or wrong-frequency data.
- Using same-day signal for same-day execution (lookahead).
- Expecting to buy the exact bottom and sell the exact top; goal is to capture most of the trend.

## See also
- `backtest_ma_crossover` (skills): implementation with store, no lookahead, basic stats
- `rolling_indicators`: safe MA computation with enough history
- `basic_concepts`: trend-following and signal states
