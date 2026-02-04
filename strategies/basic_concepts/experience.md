---
name: basic_concepts
description: Trading strategy vs capital strategy, signal states, and 策引's multi-layer strategy framework.
tags: [strategy, 交易策略, 资金策略, BUY, SELL, HOLD, EMPTY, 策引]
---

## Core rule
**Trading strategy** decides *when* to buy/sell; **capital strategy** decides *how much* to trade each time. They work together in a portfolio.

## Trading strategy vs capital strategy

### Trading strategy (交易策略)
- **Input**: price, volume, technical indicators
- **Output**: BUY, SELL, HOLD, EMPTY
- **Examples**: dual MA, chandelier exit, AI model strategy

### Capital strategy (资金策略)
- **Input**: current cash, total assets, trading signals
- **Output**: concrete trade amounts
- **Examples**: percentage strategy, fixed investment (定投), simple percentage

**Summary**: Trading strategy = "when to trade"; capital strategy = "how much to trade".

## Signal states (交易信号状态)

| State | Meaning |
|-------|--------|
| **BUY** | No position before; strategy gives an open-long signal at this node |
| **SELL** | Had position; strategy gives a close signal at this node |
| **HOLD** | In position; no close signal yet |
| **EMPTY** | No position and no open signal |

## 策引's three-layer strategy system

1. **Code strategy (代码策略)**  
   Official implementations (dual MA, momentum rotation, chandelier exit, buy-and-hold). Stateful or cross-asset logic. Users tune parameters only.

2. **Primitive strategy (原语策略)**  
   Config-based, no code. Combine indicator primitives and signal primitives (OHLC-based). "Build blocks" style. Future: AI Agent to help write/optimize.

3. **AI / LLM strategy (AI大模型策略)**  
   Multi-factor: fundamentals, technicals, news, macro. LLM + prompt produces decisions and reasoning.

## Key concepts (from doc)

- **Trend following**: Goal is "cut losses, let profits run"—not to predict turning points. Win rate may be modest; focus on long-term payoff ratio.
- **Technical indicators**: SMA/EMA, ATR, RSI (e.g. >70 overbought, <30 oversold).
- **Risk**: Stop loss, position sizing, max drawdown, Sharpe ratio.
- **Backtest**: Validate on history; be aware of overfitting and that past ≠ future.

## See also
- `dual_moving_average`, `chandelier_exit`, `momentum_rotation`: code strategies
- `primitive_strategy`: config-based strategies
- `ai_model_strategy`: LLM-based strategies
