---
name: backtest_ma_crossover
description: Backtest a simple MA crossover strategy using stock-data `store` (no external data sources), with no-lookahead signals and basic performance metrics.
tags: [backtest, 回测, strategy, 策略, MA, 均线, crossover, golden_cross, python, store]
---

## Core rule (MUST follow)
Backtests MUST use the agent’s **stock-data store** (`store.*`) as the data source.

- ✅ Use: `store.daily(...)` / `store.daily_adj(..., how="qfq")`
- ❌ Do NOT use external fetchers (AkShare/Tushare API calls) inside `tool_execute_python`
- ✅ Always sort by `trade_date` ascending before rolling or `pct_change`
- ✅ Avoid lookahead: compute signals on day *t*, execute from day *t+1* (`shift(1)`)

## Before Python: resolve to canonical `ts_code`
Backtests need a canonical code like `600519.SH` / `300888.SZ`.

- If the user provides a name or partial code, use:
  - `tool_search_stocks(query="...")` → pick the right row
  - `tool_resolve_symbol("300888")` (if needed) → get `ts_code`
- In Python, always use that `ts_code` with `store.*`

## Data access quick reminder (Tool API vs Store API)
- Tools like `tool_get_daily_prices(ts_code, limit=10)` accept `limit`
- Store methods like `store.daily(ts_code)` / `store.daily_adj(ts_code)` do **NOT** accept `limit`
  - If you need “最近N条”: `df.sort_values("trade_date", ascending=False).head(N)`

## Recommended patterns

### 1) Load data (prefer adjusted prices for returns)

```python
ts_code = "600519.SH"
start_date, end_date = "20220101", None

df = store.daily_adj(ts_code, how="qfq", start_date=start_date, end_date=end_date)
if df.empty:
    raise ValueError(f"No daily_adj data for {ts_code} since {start_date}")

need = ["trade_date", "open", "close", "high", "low", "vol"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns {missing}; got {df.columns.tolist()}")

df = df.sort_values("trade_date").reset_index(drop=True)
```

### 2) MA crossover signals (no lookahead)
- Signal computed on close of day *t*
- Trade executed on day *t+1* (approximation; avoids future leak)

```python
fast, slow = 20, 60
df["ma_fast"] = df["close"].rolling(fast).mean()
df["ma_slow"] = df["close"].rolling(slow).mean()

above = df["ma_fast"] > df["ma_slow"]
above_prev = above.shift(1)

entry_sig = above & (above_prev == False)
exit_sig = (~above) & (above_prev == True)

# Build position: 1=long, 0=flat
pos = []
state = 0
for ent, ex in zip(entry_sig.fillna(False), exit_sig.fillna(False)):
    if ent:
        state = 1
    elif ex:
        state = 0
    pos.append(state)
df["pos_raw"] = pos

# Execute next day (t+1)
df["pos"] = df["pos_raw"].shift(1).fillna(0).astype(float)
```

### 3) Returns + transaction costs (simple model)
- Use close-to-close returns for daily PnL
- Charge fee on turnover (position change); keep it simple and explicit

```python
fee_rate = 0.0003  # 3 bps per trade side (simple)

df["ret_mkt"] = df["close"].pct_change().fillna(0.0)
df["turnover"] = df["pos"].diff().abs().fillna(0.0)  # 1 when entering/exiting

df["ret_strat_gross"] = df["pos"] * df["ret_mkt"]
df["ret_strat_net"] = df["ret_strat_gross"] - df["turnover"] * fee_rate

df["equity"] = (1.0 + df["ret_strat_net"]).cumprod()
```

### 4) Minimal stats (CAGR / Sharpe / Max Drawdown)

```python
equity = df["equity"].astype(float)
ret = df["ret_strat_net"].astype(float)

days = int(ret.shape[0])
ann = 252.0

cagr = float(equity.iloc[-1] ** (ann / max(days, 1)) - 1.0)
sharpe = float((ret.mean() / (ret.std(ddof=0) + 1e-12)) * (ann ** 0.5))

peak = equity.cummax()
dd = equity / peak - 1.0
mdd = float(dd.min())

summary = {
    "ts_code": ts_code,
    "start": int(df["trade_date"].iloc[0]),
    "end": int(df["trade_date"].iloc[-1]),
    "fast": fast,
    "slow": slow,
    "CAGR(%)": round(cagr * 100, 2),
    "Sharpe": round(sharpe, 2),
    "MaxDrawdown(%)": round(mdd * 100, 2),
    "FinalEquity": round(float(equity.iloc[-1]), 4),
}

print(pd.DataFrame([summary]).to_string(index=False))
```

## Common bugs to avoid
- Using `.pct_change()` or `.rolling()` on unsorted `trade_date`.
- Forgetting `shift(1)` for execution (future leak).
- Mixing tool params with store params (store has no `limit`).
- Backtesting on unadjusted prices while claiming “total return”.

## See also
- `rolling_indicators`: how to compute MA safely (enough history)
- `adj_prices_and_returns`: why adjusted prices matter for returns
- `risk_metrics`: volatility / drawdown calculations
