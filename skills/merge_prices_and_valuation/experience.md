---
name: merge_prices_and_valuation
description: Join daily price data with daily_basic valuation data on trade_date safely.
tags: [daily, daily_basic, merge, join, pe_ttm, pb, 市值, 估值]
---

## Core rule
Prices (`store.daily`) and valuations (`store.daily_basic`) are separate tables. **Merge on `trade_date`** and re-check units.

## Recommended patterns

### Merge last N rows (aligned by trade_date)

```python
px = store.daily(ts_code, start_date="20240101").sort_values("trade_date")
vb = store.daily_basic(ts_code, start_date="20240101").sort_values("trade_date")

need_px = ["trade_date", "close", "vol", "pct_chg"]
need_vb = ["trade_date", "pe_ttm", "pb", "total_mv", "turnover_rate"]
px = px[need_px]
vb = vb[[c for c in need_vb if c in vb.columns]]

df = px.merge(vb, on="trade_date", how="inner")
result = df.tail(30)
```

## Notes
- Unit conversion/presentation details are covered in `valuation_units` (recommended to load alongside this skill).

## Common bugs to avoid
- Merging without sorting / checking duplicates (causes unexpected row counts).
- Assuming `total_mv` is already in 亿元.
- Using `how="outer"` and then forgetting to handle NaNs.

## See also
- `valuation_units`: unit conversions + presentation conventions

