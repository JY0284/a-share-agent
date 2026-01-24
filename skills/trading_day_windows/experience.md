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
- After loading data, use `pandas_trade_date` for sorting/slicing patterns and date formatting for display.

## Common bugs to avoid
- Using `.tail(N)` before sorting if you need “latest”.
- Rolling on descending dates.
- Assuming `end_date` is a trading day (it may not be).

