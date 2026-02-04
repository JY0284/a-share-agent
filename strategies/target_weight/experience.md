---
name: target_weight
description: Target-weight strategy: maintain or rebalance toward a given weight (e.g. fixed allocation or dynamic target).
tags: [strategy, 目标权重, rebalance, 策引]
---

## Core rule
Strategy defines **target weights** (e.g. 20% in asset A, 80% in cash). Trading and rebalancing are driven by deviation from these targets rather than pure BUY/SELL signals from indicators.

## Role in 策引
- **Input**: Current portfolio weights, target weights, sometimes market data.
- **Output**: Rebalance orders to move actual weights toward target (buy underweight, sell overweight).
- Often used with **capital strategy** that decides how much to move per rebalance (e.g. full rebalance vs partial).

## Typical use
- **Fixed allocation**: e.g. 60% equity / 40% bonds; rebalance when drift exceeds a threshold.
- **Dynamic target**: Target weight can change with regime or signals; strategy still outputs "target weights" and rebalancing follows.

## See also
- `basic_concepts`: trading strategy vs capital strategy
- 策引 doc: [目标权重策略](https://docs.myinvestpilot.com/docs/strategies/target-weight-strategy)
