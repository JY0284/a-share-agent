---
name: buy_and_hold
description: Buy-and-hold strategy for long-term or fixed-investment (定投) scenarios.
tags: [strategy, 买入持有, 定投, buy-and-hold, 策引]
---

## Core rule
Buy-and-hold **does not** decide timing; it assumes long-term holding. Often used with **fixed-investment (定投)** capital strategy: the strategy emits a "buy" intent every period, and the capital strategy decides how much to invest (e.g. fixed amount or percentage).

## Role in 策引
- **Trading strategy**: Produces a buy signal every period (e.g. daily); no sell signal for "hold forever" style.
- **Capital strategy**: Decides whether there is enough cash to invest and how much (e.g. FixedInvestmentStrategy).
- **Use case**: Demo and analysis for 定投; not intended as a standalone trading strategy for active portfolios.

## Important note (from doc)
This strategy is for **fixed-investment (定投) analysis demonstration**. It is not recommended for use as the sole trading strategy in a live trading portfolio.

## See also
- `basic_concepts`: trading vs capital strategy
- 策引 doc: [买入持有策略](https://docs.myinvestpilot.com/docs/strategies/buy-and-hold)
