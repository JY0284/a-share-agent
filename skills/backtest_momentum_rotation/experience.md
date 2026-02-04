---
name: backtest_momentum_rotation
description: Backtest cross-asset momentum rotation: rank symbols by N-day return, hold top-K (e.g. equal weight), rebalance at fixed frequency; stateful rotation and PnL.
tags: [backtest, 回测, momentum, rotation, multi_stock, 动量轮动, store]
---

## Core rule (MUST follow)
Backtests MUST use the agent's **stock-data store** (`store.*`).

- ✅ Use: `store.daily_adj(..., how="qfq")` for each ts_code
- ✅ Align all symbols on **same trading dates** (inner merge on trade_date)
- ✅ At each rebalance: compute N-day return per symbol, rank, assign weight to top-K (e.g. equal weight); track holdings and turnover for fees

## Strategy context
- **Input**: list of ts_codes, lookback N (e.g. 20 or 60), rebalance frequency (e.g. every 20 trading days), top-K (e.g. 1 or 2).
- **Each rebalance**: on that date, compute N-day past return for each symbol (aligned), sort descending, pick top-K; assign equal weight (1/K each).
- **Stateful**: current holdings = set of top-K; on next rebalance, new top-K may differ → trade (sell dropped, buy new), then hold until next rebalance.
- **PnL**: portfolio return = sum(weight_i * ret_i) per period; charge fee on turnover (change in weights).

## Before Python: resolve to canonical ts_codes
Use `tool_search_stocks` / `tool_resolve_symbol` if user gives names; in Python use a list of `ts_code` with `store.*`.

## Recommended patterns

### 1) Load aligned panel (all ts_codes, same dates)

```python
ts_codes = ["600519.SH", "000858.SZ", "000568.SZ"]
start_date, end_date = "20220101", "20241231"
n_days, top_k, rebal_freq = 20, 2, 20

dfs = []
for ts in ts_codes:
    df = store.daily_adj(ts, how="qfq", start_date=start_date, end_date=end_date).sort_values("trade_date")
    if df.empty:
        continue
    df = df[["trade_date", "close"]].rename(columns={"close": ts})
    dfs.append(df)
from functools import reduce
panel = reduce(lambda a, b: a.merge(b, on="trade_date", how="inner"), dfs)
panel = panel.sort_values("trade_date").reset_index(drop=True)
```

### 2) Returns per symbol; rebalance dates; at each rebalance rank and get top-K

```python
for ts in ts_codes:
    if ts not in panel.columns:
        continue
    panel[f"ret_{ts}"] = panel[ts].pct_change()
    panel[f"ret_N_{ts}"] = panel[ts].pct_change(n_days)

rebal_dates = panel["trade_date"].iloc[n_days::rebal_freq].tolist()
if not rebal_dates:
    rebal_dates = [panel["trade_date"].iloc[-1]]

weights = {ts: 0.0 for ts in ts_codes}
equity = 1.0
prev_weights = {ts: 0.0 for ts in ts_codes}
fee_rate = 0.001

for d in rebal_dates:
    row = panel[panel["trade_date"] == d]
    if row.empty:
        continue
    idx = row.index[0]
    ret_cols = [f"ret_N_{t}" for t in ts_codes if f"ret_N_{t}" in panel.columns]
    vals = [(panel.loc[idx, c] if c in panel.columns else np.nan) for c in ret_cols]
    syms = [t for t in ts_codes if f"ret_N_{t}" in panel.columns]
    ranked = sorted(zip(syms, vals), key=lambda x: (x[1] if not np.isnan(x[1]) else -np.inf), reverse=True)[:top_k]
    new_weights = {s: 1.0 / top_k for s, _ in ranked}
    for s in ts_codes:
        new_weights.setdefault(s, 0.0)
    turnover = sum(abs(new_weights.get(s, 0) - prev_weights.get(s, 0)) for s in ts_codes)
    equity *= (1.0 - turnover * fee_rate)
    prev_weights = new_weights.copy()
    weights = new_weights.copy()
```

### 3) Daily portfolio return and equity curve (simplified: use weights at last rebalance)

For a full daily curve, loop over each trading day: if day in rebal_dates update weights; then daily ret = sum(weight[ts] * ret[ts]). Omit full loop here; output summary stats from equity at rebalance dates or from a daily loop if implemented.

```python
# Simplified: daily returns using current weights (update weights only on rebalance in a loop over days)
panel["port_ret"] = 0.0
curr_weights = {ts: 0.0 for ts in ts_codes}
for i in range(len(panel)):
    t = panel["trade_date"].iloc[i]
    if t in rebal_dates:
        row = panel.iloc[i]
        ret_cols = [f"ret_N_{s}" for s in ts_codes if f"ret_N_{s}" in panel.columns]
        syms = [s for s in ts_codes if f"ret_N_{s}" in panel.columns]
        vals = [row[f"ret_N_{s}"] for s in syms]
        ranked = sorted(zip(syms, vals), key=lambda x: (x[1] if not np.isnan(x[1]) else -np.inf), reverse=True)[:top_k]
        curr_weights = {s: 1.0 / top_k for s, _ in ranked}
        for s in ts_codes:
            curr_weights.setdefault(s, 0.0)
    ret_today = sum(curr_weights.get(s, 0) * panel[f"ret_{s}"].iloc[i] if f"ret_{s}" in panel.columns and s in curr_weights else 0 for s in ts_codes)
    panel.loc[panel.index[i], "port_ret"] = ret_today if not np.isnan(ret_today) else 0.0

panel["equity"] = (1.0 + panel["port_ret"]).cumprod()
ret = panel["port_ret"].fillna(0).astype(float)
equity = panel["equity"].astype(float)
days = int(ret.shape[0])
ann = 252.0
cagr = float(equity.iloc[-1] ** (ann / max(days, 1)) - 1.0)
sharpe = float((ret.mean() / (ret.std(ddof=0) + 1e-12)) * (ann ** 0.5))
mdd = float((equity / equity.cummax() - 1.0).min())
summary = {"ts_codes": len(ts_codes), "N": n_days, "top_K": top_k, "rebal_freq": rebal_freq, "CAGR(%)": round(cagr * 100, 2), "Sharpe": round(sharpe, 2), "MaxDrawdown(%)": round(mdd * 100, 2)}
print(pd.DataFrame([summary]).to_string(index=False))
```

## Common bugs to avoid
- Comparing or ranking returns on misaligned dates (always merge on trade_date first).
- Forgetting to charge fees on weight changes at rebalance.
- Too few bars for N-day return at first rebalance (start rebal_dates after index >= n_days).

## See also
- `multi_stock_compare`: align dates, rank by momentum, top-K pattern
- `momentum_breakout`: single-asset momentum; this skill is cross-asset rotation
- `backtest_ma_crossover`: same backtest stats (CAGR, Sharpe, MDD)
