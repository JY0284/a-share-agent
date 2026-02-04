---
name: backtest_chandelier_exit
description: Backtest chandelier exit (ATR-based trailing stop): long stop = highest high − ATR×multiplier, stop only raised; exit when close < stop.
tags: [backtest, 回测, chandelier, ATR, trailing_stop, 吊灯止损, store]
---

## Core rule (MUST follow)
Backtests MUST use the agent's **stock-data store** (`store.*`).

- ✅ Use: `store.daily_adj(..., how="qfq")` with columns high, low, close
- ✅ Sort by `trade_date` ascending; compute signals on day *t*, execute from day *t+1* (`shift(1)`)

## Strategy context
- **Long stop** = (highest high over lookback) − ATR × multiplier. Stop is **only raised**, never lowered (use cummax of stop series).
- **True Range** = max(high−low, |high−prev_close|, |low−prev_close|). ATR = rolling mean of TR (e.g. period 14 or 22).
- **Exit**: when close < chandelier stop → sell. Parameters: ATR period (e.g. 22), multiplier (e.g. 3).

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
need = ["trade_date", "open", "close", "high", "low", "vol"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns {missing}")
df = df.sort_values("trade_date").reset_index(drop=True)
```

### 2) ATR and chandelier stop (stop only goes up)

```python
atr_period, mult = 22, 3
prev_close = df["close"].shift(1)
tr = np.maximum(
    df["high"] - df["low"],
    np.maximum(
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ),
)
df["atr"] = tr.rolling(atr_period).mean()
df["highest_high"] = df["high"].rolling(atr_period).max()
df["stop_raw"] = df["highest_high"] - mult * df["atr"]
df["stop"] = df["stop_raw"].cummax().ffill()
```

### 3) Position: long until close < stop; once exited stay flat (no re-entry). Execute next day

```python
exit_sig = df["close"] < df["stop"]
state = 1
pos_list = []
for ex in exit_sig.fillna(False):
    if ex and state == 1:
        state = 0
    pos_list.append(state)
df["pos_raw"] = pos_list
df["pos"] = df["pos_raw"].shift(1).fillna(0).astype(float)
```

### 4) Returns, fees, equity, stats (same as backtest_ma_crossover)

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
summary = {"ts_code": ts_code, "atr_period": atr_period, "mult": mult, "CAGR(%)": round(cagr * 100, 2), "Sharpe": round(sharpe, 2), "MaxDrawdown(%)": round(mdd * 100, 2)}
print(pd.DataFrame([summary]).to_string(index=False))
```

## Common bugs to avoid
- Forgetting cummax on stop (stop must only rise).
- Using unsorted trade_date for rolling.
- Forgetting shift(1) for execution (lookahead).

## See also
- `rolling_indicators`: ATR computation
- `risk_metrics`: volatility / drawdown
- `backtest_ma_crossover`: same backtest scaffolding (fees, equity, stats)
