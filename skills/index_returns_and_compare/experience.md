---
name: index_returns_and_compare
description: Use index_basic/index_daily correctly; compute index returns and compare indices on aligned trading dates.
tags: [index, index_basic, index_daily, 指数, 沪深300, 上证指数, 中证500, returns, compare, 对比]
---

## Core rule
Index data is **not** stock data. Use:
- `store.read("index_basic")` to discover `ts_code`
- `store.index_daily(ts_code, ...)` to load index bars

## Recommended patterns

### Discover index codes by name

```python
idx = store.read("index_basic")
hit = idx[idx["name"].astype(str).str.contains("沪深300", na=False)][["ts_code", "name", "market", "publisher"]]
hit.head(10)
```

### Compute returns for one index (use close, sorted by trade_date)

```python
df = store.index_daily("000300.SH", start_date="20230101")
df = df.sort_values("trade_date")
df["ret_1d"] = df["close"].pct_change()
df["ret_20d"] = df["close"].pct_change(20)
out = df[["trade_date", "close", "ret_1d", "ret_20d"]].dropna().tail(30)
out
```

### Compare two indices (align by trade_date)

```python
codes = {"沪深300": "000300.SH", "中证500": "000905.SH"}
frames = []
for name, code in codes.items():
    d = store.index_daily(code, start_date="20230101")[["trade_date", "close"]].copy()
    d["trade_date"] = d["trade_date"].astype(str)
    d = d.sort_values("trade_date")
    d["ret_1d"] = d["close"].pct_change()
    frames.append(d[["trade_date", "ret_1d"]].rename(columns={"ret_1d": f"{name}_ret_1d"}))

merged = frames[0]
for f in frames[1:]:
    merged = merged.merge(f, on="trade_date", how="inner")

merged = merged.dropna().tail(60)
merged
```

## Common bugs to avoid
- Comparing indices using stock tools (`store.daily`).
- Not sorting by `trade_date` before `pct_change`.
- Merging on `trade_date` with mixed dtypes (always cast to `str` first).

