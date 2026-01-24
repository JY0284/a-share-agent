---
name: adj_prices_and_returns
description: Compute returns correctly (prefer adjusted prices) and avoid common pitfalls in return presentation.
tags: [returns, pct_chg, qfq, hfq, adj, 复权, 收益率, 涨跌幅]
---

## Core rule
If you compare performance across time (returns), **prefer adjusted prices** (前复权/后复权) to reduce dividend/split distortions.

## Scope (avoid duplication)
- This skill focuses on **price type choice (复权)** and **return computation**.
- If you need to define “最近N个交易日”的窗口，请用 `trading_day_windows`.

## Recommended patterns

### Use adjusted close to compute return

```python
df = store.daily_adj(ts_code, how="qfq", start_date="20240101")
df = df.sort_values("trade_date")
df["ret_1d"] = df["close"].pct_change()
df["ret_20d"] = df["close"].pct_change(20)
result = df[["trade_date", "close", "ret_1d", "ret_20d"]].dropna().tail(20)
```

### Present returns in % with clear labeling

```python
out = result.copy()
for c in ["ret_1d", "ret_20d"]:
    out[c] = (out[c] * 100).round(2)
out = out.rename(columns={"ret_1d": "ret_1d(%)", "ret_20d": "ret_20d(%)"})
```

## Common bugs to avoid
- Mixing unadjusted prices with “total return” claims.
- Computing returns on unsorted data (always sort by `trade_date` ascending first).

## See also
- `trading_day_windows`: build windows using trading days

