---
name: backtest_macd
description: Backtest MACD crossover: MACD = fast EMA − slow EMA; signal = EMA(MACD, 9); long when MACD crosses above signal, exit when crosses below.
tags: [backtest, 回测, MACD, crossover, store]
---

## Core rule (MUST follow)
Backtests MUST use the agent's **stock-data store** (`store.*`).

- ✅ Use: `store.daily_adj(..., how="qfq")`
- ✅ Sort by `trade_date` ascending; compute signals on day *t*, execute from day *t+1* (`shift(1)`)

## Strategy context
- **MACD line** = fast EMA(12) − slow EMA(26) of close.
- **Signal line** = EMA(MACD, 9).
- **Long entry**: MACD crosses above signal (histogram turns positive).
- **Exit**: MACD crosses below signal (histogram turns negative).
- Common params: 12, 26, 9.

## Before Python: resolve to canonical `ts_code`
Use `tool_search_stocks` / `tool_resolve_symbol` if needed; in Python use that `ts_code` with `store.*`.

## Recommended patterns

### 1) Load data

```python
ts_code = "600519.SH"
start_date, end_date = "20220101", None
df = store.daily_adj(ts_code, how="qfq", start_date=start_date, end_date=end_date)
if df.empty:
    raise ValueError(f"No daily_adj data for {ts_code}")
df = df.sort_values("trade_date").reset_index(drop=True)
```

### 2) MACD and signal line; crossover signals

```python
fast, slow, signal = 12, 26, 9
df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
df["macd"] = df["ema_fast"] - df["ema_slow"]
df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
df["macd_hist"] = df["macd"] - df["macd_signal"]

above = df["macd"] > df["macd_signal"]
above_prev = above.shift(1)
entry_sig = above & (above_prev == False)
exit_sig = (~above) & (above_prev == True)

pos = []
state = 0
for ent, ex in zip(entry_sig.fillna(False), exit_sig.fillna(False)):
    if ent:
        state = 1
    elif ex:
        state = 0
    pos.append(state)
df["pos_raw"] = pos
df["pos"] = df["pos_raw"].shift(1).fillna(0).astype(float)
```

### 3) Returns, fees, equity, stats (same as backtest_ma_crossover)

```python
fee_rate = 0.0003
df["ret_mkt"] = df["close"].pct_change().fillna(0.0)
df["turnover"] = df["pos"].diff().abs().fillna(0.0)
df["ret_strat_net"] = df["pos"] * df["ret_mkt"] - df["turnover"] * fee_rate
df["equity"] = (1.0 + df["ret_strat_net"]).cumprod()
equity = df["equity"].astype(float)
ret = df["ret_strat_net"].astype(float)
days = int(ret.shape[0])
ann = 252.0
cagr = float(equity.iloc[-1] ** (ann / max(days, 1)) - 1.0)
sharpe = float((ret.mean() / (ret.std(ddof=0) + 1e-12)) * (ann ** 0.5))
peak = equity.cummax()
mdd = float((equity / peak - 1.0).min())
summary = {"ts_code": ts_code, "fast": fast, "slow": slow, "signal": signal, "CAGR(%)": round(cagr * 100, 2), "Sharpe": round(sharpe, 2), "MaxDrawdown(%)": round(mdd * 100, 2)}
print(pd.DataFrame([summary]).to_string(index=False))
```

## Common bugs to avoid
- Using unsorted trade_date for ewm/rolling.
- Forgetting shift(1) for execution (lookahead).
- Too few rows for slow EMA (pull 60+ bars).

## See also
- `rolling_indicators`: MACD computation
- `backtest_ma_crossover`: same backtest scaffolding (fees, equity, stats)
