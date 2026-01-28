---
name: etf_nav_and_premium
description: Use fund_basic/etf_daily/fund_nav to analyze ETF NAV, returns, and (if available) price-vs-NAV deviations.
tags: [etf, fund_basic, etf_daily, fund_nav, nav, premium, discount, 份额, 净值, ETF]
---

## Core rule
ETF analysis often needs **two time series**:
- **Market price bars**: `store.read("etf_daily", where={"ts_code": ...})`
- **NAV**: `store.fund_nav(ts_code, ...)`

Only compute “premium/discount” if the columns you need are actually present in your stored datasets.

## Recommended patterns

### Discover ETF codes by name

```python
fb = store.read("fund_basic")
hit = fb[fb["name"].astype(str).str.contains("沪深300", na=False)][["ts_code", "name", "management", "fund_type"]]
hit.head(10)
```

### Compute ETF price returns from `etf_daily`

```python
px = store.read("etf_daily", where={"ts_code": "510300.SH"}, start_date="20230101")
px = px.sort_values("trade_date")
px["ret_20d"] = px["close"].pct_change(20)
px[["trade_date", "close", "ret_20d"]].dropna().tail(20)
```

### Join price and NAV on date (best-effort)

```python
px = store.read("etf_daily", where={"ts_code": "510300.SH"}, start_date="20230101")[["trade_date", "close"]].copy()
px["trade_date"] = px["trade_date"].astype(str)
px = px.sort_values("trade_date")

nav = store.fund_nav("510300.SH", start_date="20230101")
# nav_date is typically the NAV date column
nav["nav_date"] = nav["nav_date"].astype(str)

merged = px.merge(nav, left_on="trade_date", right_on="nav_date", how="inner")

# If unit_nav exists, compute price-to-nav ratio (not always 1:1 units; interpret carefully)
if "unit_nav" in merged.columns:
    merged["price_to_unit_nav"] = merged["close"] / merged["unit_nav"]

merged.tail(20)
```

## Common bugs to avoid
- Assuming NAV has the same calendar as trading days (it often does, but confirm).
- Assuming premium/discount can be computed without the right NAV fields.
- Forgetting to sort before `pct_change`.

