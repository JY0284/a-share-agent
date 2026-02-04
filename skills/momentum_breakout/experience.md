---
name: momentum_breakout
description: Simple momentum and breakout signals using rolling highs/lows and return ranks.
tags: [momentum, breakout, high52w, donchian, 动量, 突破, 新高]
---

## Core rule
Momentum/breakout signals are **window-based**; use enough history and avoid lookahead by using `.shift(1)` when comparing to “prior highs”.

## Scope
This skill is **single-asset** (one ts_code): breakout, 52w high, N-day momentum. For **cross-asset momentum rotation** (rank many symbols, hold top-K, rebalance over time), use multi-symbol ranking plus a rebalance backtest; that pattern is in a separate backtest skill.

## Recommended patterns

### 20-day breakout (close > prior 20D high)

```python
df = store.daily(ts_code, start_date="20240101").sort_values("trade_date")
prior_high20 = df["high"].rolling(20).max().shift(1)
df["breakout20"] = df["close"] > prior_high20
result = df[df["breakout20"]][["trade_date", "close"]].tail(20)
```

### 52-week high proximity (roughly 252 trading days)

```python
df = store.daily_adj(ts_code, how="qfq", start_date="20230101").sort_values("trade_date")
high252 = df["close"].rolling(252).max()
df["high52w_ratio"] = df["close"] / high252
result = df[["trade_date", "close", "high52w_ratio"]].dropna().tail(20)
```

### Momentum return over N days (e.g., 60D)

```python
df["mom60(%)"] = (df["close"].pct_change(60) * 100).round(2)
result = df[["trade_date", "close", "mom60(%)"]].dropna().tail(20)
```

## Common bugs to avoid
- Using current-window high without shift (introduces lookahead in “breakout” definition).
- Too short lookback → mostly NaNs or noisy signals.
- Using unadjusted prices for long-window momentum.

## See also
- `rolling_indicators`: MA/RSI style indicators (different from breakout signals)
- `adj_prices_and_returns`: adjusted-price conventions for long-window momentum

