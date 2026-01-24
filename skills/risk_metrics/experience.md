---
name: risk_metrics
description: Compute volatility, max drawdown, and simple risk summaries from (adjusted) close series.
tags: [risk, volatility, drawdown, 回撤, 波动率, 风险]
---

## Core rule
Risk metrics require **chronological prices** and **returns**. Prefer adjusted close when measuring multi-week/month risk.

## Recommended patterns

### Realized volatility (annualized) from daily returns

```python
df = store.daily_adj(ts_code, how="qfq", start_date="20240101").sort_values("trade_date")
df["ret"] = df["close"].pct_change()
vol_daily = df["ret"].std(skipna=True)
vol_ann = float(vol_daily * np.sqrt(252))
result = {"ts_code": ts_code, "vol_ann": vol_ann}
```

### Max drawdown (MDD)

```python
df = store.daily_adj(ts_code, how="qfq", start_date="20240101").sort_values("trade_date")
price = df["close"].astype(float)
peak = price.cummax()
dd = price / peak - 1.0
mdd = float(dd.min())
last = df.tail(1).iloc[0]
result = {"ts_code": ts_code, "asof": int(last["trade_date"]), "max_drawdown(%)": round(mdd * 100, 2)}
```

### Rolling volatility (e.g., 20D)

```python
df["vol20_ann"] = df["ret"].rolling(20).std() * np.sqrt(252)
result = df[["trade_date", "vol20_ann"]].dropna().tail(20)
```

## Common bugs to avoid
- Computing drawdown on unsorted dates (must be ascending).
- Using price std dev instead of return std dev.
- Forgetting to annualize assumptions (252 trading days is a convention).

