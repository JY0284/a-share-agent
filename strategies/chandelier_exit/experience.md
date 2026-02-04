---
name: chandelier_exit
description: ATR-based trailing stop; long stop = highest high minus ATR×multiplier, only moves up.
tags: [strategy, 吊灯止损, ATR, trailing stop, 策引]
---

## Core rule
- **Long**: Stop price = (highest high over lookback) − ATR × multiplier. Stop is **only raised**, never lowered.
- **Short** (if applicable): Stop = (lowest low) + ATR × multiplier; only lowered.
- When price hits stop → **SELL** (or close long). Combines trend following with dynamic risk.

## Formula (from doc)
- True range: `max(high−low, |high−prev_close|, |low−prev_close|)`
- ATR = N-period (e.g. 14 or 22) average of true range
- Chandelier long stop = highest high over period − ATR × multiplier (often **3**)

## Parameters (from 策引 / common practice)
- **ATR period**: 14 (short), 22 (mid, common), 50 (long).
- **Multiplier**: 2 (tighter, more whipsaw), **3** (common), 4 (looser, larger drawdown).
- **Lookback for highest high**: Often same as ATR period or 22.

## Why it works
- **Volatility-adjusted**: High volatility → wider stop; low → tighter.
- **Locks in profit**: As price rises, highest high rises → stop rises.
- **Rule-based**: Reduces subjective stop placement and emotion.

## When it fits
- Clear **uptrend**; good for holding through pullbacks without giving back too much.
- **Not** ideal in choppy or mean-reverting markets (can get stopped out often).

## See also
- `basic_concepts`: trend following, stop loss
- `risk_metrics`: volatility / ATR context
- 策引 doc: [吊灯止损策略](https://docs.myinvestpilot.com/docs/strategies/chandelier-exit)
