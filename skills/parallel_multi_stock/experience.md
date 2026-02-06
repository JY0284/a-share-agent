---
name: parallel_multi_stock
description: Patterns to handle MANY stocks efficiently in ONE Python run (batch + optional safe parallel load), with aligned dates and robust empty-data handling.
tags: [parallel, batch, multi-stock, concurrency, performance, 并行, 批量, 多股票, 提速]
---

## Goal
When the task involves **many symbols** (e.g., 50–500 stocks) and you must compute the *same* metrics (returns, vol, rank, factor, signals), do it in **one `tool_execute_python` call** using **batch + vectorized groupby**.

## Core rules (must follow)
1. **One Python run for many stocks**: avoid calling per-stock tools in a loop.
2. **Align by trading dates**: sort by `ts_code, trade_date`; compute returns within each `ts_code`.
3. **Minimize IO + memory**: select only needed columns; optionally process `ts_codes` in chunks.
4. **Robust output**: handle empty frames; report how many symbols have data.

---

## Pattern A (recommended): batch → concat → groupby (fast + simple)
Compute cross-sectional metrics for many stocks, then take the latest row per stock.

```python
import pandas as pd

# Inputs
ts_codes = [
    "000001.SZ",
    "600519.SH",
    "000858.SZ",
]
start_date, end_date = "20240102", "20240205"

frames = []
for ts in ts_codes:
    df = store.daily_adj(ts, how="qfq", start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        continue

    # Ensure ts_code exists (some stores may not include it)
    if "ts_code" not in df.columns:
        df = df.assign(ts_code=ts)

    # Keep only what you need
    frames.append(df[["ts_code", "trade_date", "close"]])

all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ts_code", "trade_date", "close"])

if all_df.empty:
    result = {
        "count_input": len(ts_codes),
        "count_with_data": 0,
        "table": pd.DataFrame(),
    }
else:
    all_df = all_df.sort_values(["ts_code", "trade_date"])

    # Example metrics
    all_df["ret_5d"] = all_df.groupby("ts_code")["close"].pct_change(5)
    all_df["ret_1d"] = all_df.groupby("ts_code")["close"].pct_change(1)

    # Take latest row per symbol
    last = all_df.groupby("ts_code").tail(1)
    table = last[["ts_code", "trade_date", "close", "ret_1d", "ret_5d"]].copy()

    # Rank (descending)
    table = table.sort_values("ret_5d", ascending=False)

    result = {
        "count_input": len(ts_codes),
        "count_with_data": int(table["ts_code"].nunique()),
        "table": table,
    }

# In the sandbox, return via `result` (dict/DataFrame are formatted automatically)
result = result
```

**Why this works well**
- IO per symbol (if no bulk API) but **computation is vectorized**.
- Clear and robust: empty data won’t crash the run.

---

## Pattern B: chunked batch loading (stable + scalable)
When the universe is very large (e.g., 300–2000 symbols), **chunk** to reduce peak memory and avoid timeouts.

```python
import pandas as pd

def chunks(xs, n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]

ts_codes = ["000001.SZ", "600519.SH", "000858.SZ"]  # replace with your big universe
start_date, end_date = "20240102", "20240205"
chunk_size = 200

tables = []
for batch in chunks(ts_codes, chunk_size):
    frames = []
    for ts in batch:
        df = store.daily_adj(ts, how="qfq", start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            continue
        if "ts_code" not in df.columns:
            df = df.assign(ts_code=ts)
        frames.append(df[["ts_code", "trade_date", "close"]])

    if not frames:
        continue

    all_df = pd.concat(frames, ignore_index=True).sort_values(["ts_code", "trade_date"])
    all_df["ret_5d"] = all_df.groupby("ts_code")["close"].pct_change(5)
    table = all_df.groupby("ts_code").tail(1)[["ts_code", "trade_date", "close", "ret_5d"]]
    tables.append(table)

final = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=["ts_code", "trade_date", "close", "ret_5d"])
final = final.sort_values("ret_5d", ascending=False)
result = {"count_input": len(ts_codes), "count_with_data": int(final["ts_code"].nunique()), "table": final}
result = result
```

---

## Practical tips
- For large universes, **chunk**: `for chunk in chunks(ts_codes, 100): ...` then concat results.
- Prefer adjusted prices (`daily_adj`, `how="qfq"`) for returns/ranking.
- Always `sort_values(["ts_code","trade_date"])` before `pct_change/rolling`.

## Common pitfalls (from traces)
- Don’t use `store.read(..., offset=...)`: `StockStore.read()` does not support `offset` pagination. If you need pagination, call `tool_get_universe(offset, limit)` first, then slice `ts_codes` and load in chunks.
- Normalize date dtypes early: some datasets store `trade_date` as string; cast to YYYYMMDD int before doing comparisons like `>= 20240101`.

## See also
- `multi_stock_compare`: multi-stock alignment and ranking
- `trading_day_windows`: build windows by trading days
- `adj_prices_and_returns`: return conventions and adjusted-price usage
