---
name: momentum_rotation
description: Rotate capital into assets with strongest recent momentum; cross-asset comparison and stateful logic.
tags: [strategy, 动量轮动, rotation, 策引]
---

## Core rule
At each rebalance, **rank** all candidate assets by a momentum measure (e.g. past N-day return or other momentum score), then allocate to the **top** name(s). When another asset’s momentum overtakes the current one, **rotate** (sell current, buy new). This is **stateful** and **cross-asset**, so it is implemented as a **code strategy** in 策引.

## How it works (from doc)
- Evaluate **historical price momentum** (e.g. performance over a lookback) for each candidate.
- Allocate to the strongest performer(s).
- When a different asset becomes stronger → reduce/exit current position and add the new leader.
- Suited to **multi-asset / multi-ETF** themes (e.g. sector or region rotation).

## Why code strategy
- Needs **cross-asset comparison** and **state** (who is currently held, when to rotate).
- Not expressible as a single-asset OHLC indicator; 策引 implements it as an official code strategy.

## Typical use
- ETF rotation (e.g. by sector or region).
- Multi-stock pools: hold top-K by momentum, rebalance periodically.

## See also
- `basic_concepts`: code strategy vs primitive vs AI
- `dual_moving_average`: single-asset trend following
- 策引 doc: [动量轮动策略](https://docs.myinvestpilot.com/docs/strategies/momentum-rotation)
