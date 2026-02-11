````markdown
---
name: etf_data
description: Correct API patterns for loading ETF/fund data from the store. ETFs don't have adj_factor - use raw prices.
tags: [etf, fund, etf_daily, fund_nav, fund_basic, 基金, ETF, 场内基金]
---

## Core rule
**ETFs do NOT have 复权因子 (adj_factor)** - they don't split like stocks.

- For **stocks**: use `store.daily_adj(ts_code, how="hfq")` for backward-adjusted prices
- For **ETFs**: use `store.etf_daily(ts_code)` directly - no adjustment needed

## Available ETF data methods

| Method | Description | Example |
|--------|-------------|---------|
| `store.read("fund_basic")` | All fund metadata | Discover ETF codes |
| `store.etf_daily(ts_code, ...)` | ETF daily OHLCV bars | Price analysis |
| `store.fund_nav(ts_code, ...)` | Net Asset Value time series | NAV tracking |
| `store.fund_share(ts_code, ...)` | Share outstanding changes | Fund flows |
| `store.fund_div(ts_code)` | Dividend history | Distribution analysis |

## Recommended patterns

### Discover ETF codes by name

```python
fb = store.read("fund_basic")
hit = fb[fb["name"].astype(str).str.contains("沪深300", na=False)][["ts_code", "name", "management", "fund_type"]]
hit.head(10)
```

### Load ETF daily prices (NO adj needed)

```python
# ✅ CORRECT: Use store.etf_daily() method
df = store.etf_daily("510300.SH", start_date="20230101")
if df.empty:
    raise ValueError("No etf_daily data for 510300.SH")
df = df.sort_values("trade_date").reset_index(drop=True)

# ❌ WRONG: Don't try to use daily_adj for ETFs
# df = store.daily_adj("510300.SH", how="hfq")  # This won't work for ETFs!
```

### Backtest ETF strategy

```python
ts_code = "510300.SH"
df = store.etf_daily(ts_code, start_date="20200101")
if df.empty:
    raise ValueError(f"No ETF data for {ts_code}")

# Sort and ensure date is int for comparisons
df = df.sort_values("trade_date").reset_index(drop=True)
df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").astype(int)

# Calculate returns (no adj needed for ETF)
df["ret"] = df["close"].pct_change()
df["cum_ret"] = (1 + df["ret"]).cumprod() - 1

print(f"ETF {ts_code} cumulative return: {df['cum_ret'].iloc[-1]:.2%}")
```

### Load NAV data (if available)

```python
nav = store.fund_nav("510300.SH", start_date="20230101")
if nav.empty:
    print("No fund_nav data available; use etf_daily prices instead")
else:
    nav = nav.sort_values("nav_date")
    print(nav[["nav_date", "unit_nav", "accum_nav"]].tail(10))
```

## ⚠️ Date comparison (CRITICAL)

Date columns are often strings. **Normalize before comparing with int:**

```python
# WRONG - TypeError if trade_date is string
df[df["trade_date"] >= 20240101]  # ❌

# RIGHT - convert to int first
df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").astype(int)
df[df["trade_date"] >= 20240101]  # ✅
```

## Common bugs to avoid
- Calling `store.daily_adj()` for ETFs (ETFs don't have adj_factor)
- Assuming all funds have NAV data (check if `df.empty` after loading)
- Comparing string trade_date with int literals
- Using `store.read("etf_daily", where={...})` instead of `store.etf_daily(ts_code)`

## See also
- `etf_nav_and_premium`: NAV and premium/discount analysis
- `adj_prices_and_returns`: why stocks need 复权 but ETFs don't
- `backtest_ma_crossover`: backtesting patterns (use hfq for stocks, raw for ETFs)
````
