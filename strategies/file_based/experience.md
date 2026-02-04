---
name: file_based_strategy
description: Strategy driven by external trade records (e.g. file); used for replay, audit, or linking external execution to 策引.
tags: [strategy, 交易记录, file, 策引]
---

## Core rule
**File-based (trading record) strategy** does not compute signals from market data inside 策引; it **reads trade records** from an external source (e.g. file). Used to replay historical trades, align external execution with 策引’s portfolio view, or audit actual trades.

## Role in 策引
- **Input**: File or feed of trades (e.g. date, symbol, side, size, price).
- **Output**: Same signal states (BUY/SELL/HOLD/EMPTY) or position series implied by those trades, so 策引 can attribute PnL, risk, and performance to "this strategy."

## Typical use
- Import broker or spreadsheet trade history.
- Backtest or report on "what would have happened if we had followed this record."
- Compliance or record-keeping: strategy = "what was actually traded."

## See also
- `basic_concepts`: trading strategy as signal generator
- 策引 doc: [交易记录策略](https://docs.myinvestpilot.com/docs/strategies/file-based-strategy)
