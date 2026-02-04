---
name: backtest_bollinger
description: Backtest Bollinger Bands (mean reversion): middle = SMA(period), upper/lower = middle ± k×std; buy when price at/below lower band, sell at upper band or middle.
tags: [backtest, 回测, Bollinger, 布林带, mean_reversion, store]
---

## Core rule (MUST follow)
Backtests MUST use the agent's **stock-data store** (`store.*`).

- ✅ Use: `store.daily_adj(..., how="qfq")`
- ✅ Sort by `trade_date` ascending; compute signals on day *t*, execute from day *t+1* (`shift(1)`)

## Strategy context
- **Middle** = SMA(period), e.g. 20. **Upper** = middle + k×rolling_std, **Lower** = middle − k×rolling_std (k=2 typical).
- **Mean reversion**: buy when price touches or goes below lower band; sell when price touches upper band or returns to middle.
- **Breakout** variant: buy when close above upper band (different logic). This skill uses mean reversion.

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

### 2) Bollinger Bands and mean-reversion signals

```python
period, k = 20, 2
df["bb_mid"] = df["close"].rolling(period).mean()
std = df["close"].rolling(period).std()
df["bb_upper"] = df["bb_mid"] + k * std
df["bb_lower"] = df["bb_mid"] - k * std

# Mean reversion: buy when close <= lower, sell when close >= upper or >= mid
at_lower = df["close"] <= df["bb_lower"]
at_upper = df["close"] >= df["bb_upper"]
at_mid = df["close"] >= df["bb_mid"]
exit_sig = at_upper | at_mid

pos = []
state = 0
for al, ex in zip(at_lower.fillna(False), exit_sig.fillna(False)):
    if al:
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
summary = {"ts_code": ts_code, "period": period, "k": k, "CAGR(%)": round(cagr * 100, 2), "Sharpe": round(sharpe, 2), "MaxDrawdown(%)": round(mdd * 100, 2)}
print(pd.DataFrame([summary]).to_string(index=False))
```

## Common bugs to avoid
- Using unsorted trade_date for rolling.
- Forgetting shift(1) for execution (lookahead).
- NaN in first period-1 rows for middle/std; dropna or align index before position loop.

## See also
- `rolling_indicators`: Bollinger Bands computation
- `risk_metrics`: volatility
- `backtest_ma_crossover`: same backtest scaffolding (fees, equity, stats)
