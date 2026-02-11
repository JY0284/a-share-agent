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

## ⚠️ Date comparison (CRITICAL)
Date columns (`trade_date`, `end_date`) are often strings. **Always convert to int before comparing with int literals.**

```python
# WRONG - comparing string to int raises TypeError
df[df["trade_date"] >= 20240101]  # ❌ TypeError if trade_date is string

# RIGHT - ensure both are same type
df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").astype(int)
df[df["trade_date"] >= 20240101]  # ✅ Works correctly

# OR use string comparison (also works)
df[df["trade_date"].astype(str) >= "20240101"]  # ✅ String comparison
```

## ETF/Fund adjusted prices (复权)
**ETFs do NOT have 复权因子 (adj_factor)**. They don't split like stocks.

- For **stocks**: use `store.daily_adj(ts_code, how="hfq")` for backward-adjusted prices
- For **ETFs**: use raw prices from `store.etf_daily(ts_code)` directly
  - ETF prices are already "clean" (no splits to adjust for)
  - If you need NAV-based returns, use `store.fund_nav(ts_code)` instead

```python
# Stock: use 后复权 (hfq) for backtesting
df_stock = store.daily_adj("600519.SH", how="hfq", start_date="20200101")

# ETF: use etf_daily directly (no adj needed)
df_etf = store.etf_daily("510300.SH", start_date="20200101")
```

## Common bugs to avoid
- Mixing unadjusted prices with "total return" claims.
- Computing returns on unsorted data (always sort by `trade_date` ascending first).
- **Comparing string dates with int literals** (always normalize types first).
- Calling `store.daily_adj()` for ETFs (use `store.etf_daily()` instead).

## See also
- `trading_day_windows`: build windows using trading days
- `etf_nav_and_premium`: ETF-specific data access patterns

