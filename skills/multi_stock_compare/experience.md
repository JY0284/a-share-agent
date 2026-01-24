---
name: multi_stock_compare
description: Compare multiple stocks on the same metric/date window with aligned returns and clear units.
tags: [compare, multi-stock, returns, correlation, 对比, 比较, 多股票]
---

## Core rule
When comparing multiple stocks, **align on the same trading dates** and use consistent price types (prefer adjusted).

## Recommended patterns

### Compare recent returns across a list of ts_codes

```python
ts_codes = ["600519.SH", "000858.SZ", "000568.SZ"]
start_date, end_date = "20241001", "20250110"  # if you need “last N trading days”, see `trading_day_windows`

rows = []
for ts in ts_codes:
    df = store.daily_adj(ts, how="qfq", start_date=start_date, end_date=end_date).sort_values("trade_date")
    if df.empty:
        continue
    df["ret_20d"] = df["close"].pct_change(20)
    last = df.dropna().tail(1)
    if last.empty:
        continue
    rows.append({"ts_code": ts, "trade_date": int(last["trade_date"].iloc[0]), "ret_20d(%)": round(float(last["ret_20d"].iloc[0] * 100), 2)})

result = pd.DataFrame(rows).sort_values("ret_20d(%)", ascending=False)
```

### Correlation of daily returns between two stocks

```python
a, b = "600519.SH", "000858.SZ"
start_date, end_date = "20241001", "20250110"  # if you need “last N trading days”, see `trading_day_windows`
df_a = store.daily_adj(a, how="qfq", start_date=start_date, end_date=end_date).sort_values("trade_date")[["trade_date", "close"]]
df_b = store.daily_adj(b, how="qfq", start_date=start_date, end_date=end_date).sort_values("trade_date")[["trade_date", "close"]]
df = df_a.merge(df_b, on="trade_date", how="inner", suffixes=("_a", "_b"))
df["ret_a"] = df["close_a"].pct_change()
df["ret_b"] = df["close_b"].pct_change()
result = {"pair": f"{a} vs {b}", "corr_ret": float(df[["ret_a", "ret_b"]].dropna().corr().iloc[0, 1])}
```

## Common bugs to avoid
- Comparing returns computed on different date ranges.
- Forgetting to use adjusted prices for performance comparisons.
- Presenting incomparable units (e.g., mix % and raw numbers).

## See also
- `trading_day_windows`: build windows using trading days
- `adj_prices_and_returns`: return computation + adjusted-price conventions

