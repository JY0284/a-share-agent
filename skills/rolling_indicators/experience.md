---
name: rolling_indicators
description: Computing MA/EMA/RSI, ATR, MACD, Bollinger Bands and ensuring enough lookback data.
tags: [pandas, MA, EMA, RSI, ATR, MACD, Bollinger, rolling]
---

## Core rule
Rolling indicators require enough history. Pull at least:
- MA60 → 120+ rows is safer
- RSI14 → 60+ rows is safer
- ATR22 → 60+ rows is safer
- MACD(12,26,9) → 60+ rows is safer
- Bollinger(20) → 40+ rows is safer

## Robust template (avoid common KeyError/empty-data bugs)

Use this skeleton before any rolling/EMA logic:

```python
df = store.daily(ts_code, start_date="20240101")

required = {"trade_date", "close"}
if df is None or df.empty or not required.issubset(df.columns):
    result = {"ts_code": ts_code, "ok": False, "reason": "no data or missing cols", "cols": list(df.columns) if df is not None else None}
else:
    df = df.sort_values("trade_date")
    # Optional: normalize trade_date dtype for comparisons/slicing
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").astype(int)
    result = {"ts_code": ts_code, "ok": True, "n_rows": int(len(df))}
```

## Moving averages (MA)

```python
df = store.daily(ts_code, start_date="20240101")
if df is None or df.empty:
    result = pd.DataFrame()
else:
    df = df.sort_values("trade_date")
df["ma5"] = df["close"].rolling(5).mean()
df["ma20"] = df["close"].rolling(20).mean()
df["ma60"] = df["close"].rolling(60).mean()
result = df[["trade_date", "close", "ma5", "ma20", "ma60"]].dropna().tail(20)
```

## RSI (14)

```python
df = store.daily(ts_code, start_date="20240101").sort_values("trade_date")
delta = df["close"].diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
df["rsi14"] = 100 - (100 / (1 + rs))
result = df[["trade_date", "close", "rsi14"]].dropna().tail(20)
```

## ATR (Average True Range)

True Range = max(high−low, |high−prev_close|, |low−prev_close|). ATR = rolling mean of TR. Used in chandelier/trailing-stop strategies (volatility-adjusted stops).

```python
df = store.daily(ts_code, start_date="20240101").sort_values("trade_date")
prev_close = df["close"].shift(1)
tr = np.maximum(
    df["high"] - df["low"],
    np.maximum(
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ),
)
period = 22  # or 14 for shorter
df["atr"] = tr.rolling(period).mean()
result = df[["trade_date", "close", "atr"]].dropna().tail(20)
```

## MACD

MACD line = fast EMA − slow EMA; signal line = EMA of MACD (e.g. 9). Crossover above signal → common long entry; crossover below → common exit.

```python
df = store.daily(ts_code, start_date="20240101").sort_values("trade_date")
fast, slow, signal = 12, 26, 9
df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
df["macd"] = df["ema_fast"] - df["ema_slow"]
df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
df["macd_hist"] = df["macd"] - df["macd_signal"]
result = df[["trade_date", "close", "macd", "macd_signal", "macd_hist"]].dropna().tail(20)
```

## Bollinger Bands

Middle = SMA(period); upper = middle + k×rolling_std; lower = middle − k×rolling_std. Mean reversion: buy near lower band, sell near upper/middle. Breakout: buy when close above upper band (often with volume).

```python
df = store.daily(ts_code, start_date="20240101").sort_values("trade_date")
period, k = 20, 2
df["bb_mid"] = df["close"].rolling(period).mean()
std = df["close"].rolling(period).std()
df["bb_upper"] = df["bb_mid"] + k * std
df["bb_lower"] = df["bb_mid"] - k * std
result = df[["trade_date", "close", "bb_mid", "bb_upper", "bb_lower"]].dropna().tail(20)
```

## Common bugs to avoid
- Rolling on unsorted data.
- Using too short a window, leading to mostly NaN outputs.
- Div-by-zero in RSI (handle avg_loss==0).
- ATR: need high, low, and prev close; sort by trade_date first.
- MACD: need enough bars for slow EMA (e.g. 60+ for 26-period).
- Bollinger: std can be NaN for first period-1 rows; dropna or fill.

