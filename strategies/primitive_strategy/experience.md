---
name: primitive_strategy
description: Config-based strategies built from indicator and signal primitives—no code; "building blocks" for OHLC-based logic.
tags: [strategy, 原语策略, primitive, config, 策引]
---

## Core rule
**Primitive strategy** = configuration (e.g. JSON/DSL), not code. You combine **indicator primitives** (e.g. RSI, MACD, MA) and **signal primitives** (buy/sell/hold logic) like building blocks. Suited to "when condition A and B both hold, buy" style rules on OHLC data.

## Role in 策引
- **Indicator primitives**: RSI, MACD, moving averages, etc., on OHLC.
- **Signal primitives**: Logic that maps indicator outputs to BUY/SELL/HOLD/EMPTY.
- **Config**: Compose indicators and signals into one strategy; no programming required.
- **Future**: AI Agent may help users write and optimize primitive strategy configs.

## Strengths
- **No code**: Good for users with clear logic but no coding.
- **Transparent**: Each block’s role is clear; easy to debug and explain.
- **Fast iteration**: Change config and re-run backtest.

## Limits (from doc)
- Best for **single-asset, OHLC-based** logic without complex state.
- **Not** for cross-asset comparison or heavy state (use code strategy instead).
- Complex chart patterns or custom state are not in scope.

## When to use
- Classic technical indicator combos (e.g. RSI + MA, MACD + volume).
- Multi-condition rules: "RSI < 30 and price above 50d MA → BUY".
- Quick prototyping and backtesting of rule-based ideas.

## See also
- `basic_concepts`: three-layer system (code vs primitive vs AI)
- 策引: [原语策略](https://docs.myinvestpilot.com/docs/strategies/primitive-strategy), [原语系统架构](https://docs.myinvestpilot.com/docs/primitives/architecture), [指标原语](https://docs.myinvestpilot.com/docs/primitives/indicators)
