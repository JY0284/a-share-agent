---
name: rolling_indicators
description: Computing MA/EMA/RSI and ensuring enough lookback data.
tags: [pandas, MA, EMA, RSI, rolling]
---

## Core rule
Rolling indicators require enough history. Pull at least:
- MA60 → 120+ rows is safer
- RSI14 → 60+ rows is safer

## Moving averages (MA)

```python
df = store.daily(ts_code, start_date="20240101").sort_values("trade_date")
df["ma5"] = df["close"].rolling(5).mean()
df["ma20"] = df["close"].rolling(20).mean()
df["ma60"] = df["close"].rolling(60).mean()
result = df[["trade_date", "close", "ma5", "ma20", "ma60"]].dropna().tail(20)
```

## RSI (14)

```python
df = store.daily(ts_code, start_date="20240101").sort_values("trade_date")
delta = df["close"].diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
df["rsi14"] = 100 - (100 / (1 + rs))
result = df[["trade_date", "close", "rsi14"]].dropna().tail(20)
```

## Common bugs to avoid
- Rolling on unsorted data.
- Using too short a window, leading to mostly NaN outputs.
- Div-by-zero in RSI (handle avg_loss==0).

