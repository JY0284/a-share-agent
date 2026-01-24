---
name: pandas_trade_date
description: Safe patterns for sorting by trade_date and selecting “recent N trading days”.
tags: [pandas, trade_date, sorting, recent]
---

## Core rule
Always sort explicitly on `trade_date` and be clear about direction.

## Recommended patterns

### Get most recent N rows

```python
df = store.daily(ts_code)
df = df.sort_values("trade_date", ascending=False)
recent = df.head(N)
```

### Get last N rows in chronological order (for rolling indicators)

```python
df = store.daily(ts_code, start_date="20240101")
df = df.sort_values("trade_date", ascending=True)
tail = df.tail(N)  # still chronological
```

### Convert dates for display

```python
out = recent.copy()
out["trade_date"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
```

## Common bugs to avoid
- Don’t assume store returns sorted data.
- Don’t call `tail(N)` before sorting if you need “latest”.
- If doing rolling features (MA/RSI), sort ascending before `rolling(...)`.

