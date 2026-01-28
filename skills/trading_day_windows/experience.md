---
name: trading_day_windows
description: Build robust date ranges using trading days (not calendar days) and keep date direction consistent.
tags: [trade_date, trading_days, window, 日期, 交易日, recent]
---

## Core rule
Use **trading days** to define lookbacks (“最近N个交易日”), not calendar offsets.

## Recommended patterns

### Last N trading days ending at an end_date

```python
end_date = "20250110"
days = store.trading_days("20000101", end_date)
if len(days) < 60:
    raise ValueError("Not enough trading days in calendar")
start_date = str(days[-60])
df = store.daily(ts_code, start_date=start_date, end_date=end_date).sort_values("trade_date")
result = df[["trade_date", "close", "vol", "pct_chg"]]
```

## Notes
- Always sort explicitly on `trade_date` and be clear about direction:
  - Rolling/indicators: sort **ascending** then `rolling(...)`
  - “最新N条”: sort **descending** then `head(N)`

### Get most recent N rows (for display)

```python
df = store.daily(ts_code)
df = df.sort_values("trade_date", ascending=False)
recent = df.head(N)
```

### Get last N rows in chronological order (for rolling indicators)

```python
df = store.daily(ts_code, start_date=start_date, end_date=end_date)
df = df.sort_values("trade_date", ascending=True)
tail = df.tail(N)  # still chronological
```

### Convert trade_date for display (YYYY-MM-DD)

```python
out = recent.copy()
out["trade_date"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
```

## Common bugs to avoid
- Using `.tail(N)` before sorting if you need “latest”.
- Rolling on descending dates.
- Assuming `end_date` is a trading day (it may not be).

