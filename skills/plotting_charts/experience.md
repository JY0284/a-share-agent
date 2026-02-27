---
name: plotting_charts
description: Creating matplotlib charts for stock analysis - price trends, backtest equity curves, comparisons, with proper Chinese font support and automatic figure capture.
tags: [matplotlib, plt, chart, plot, visualization, 图表, 走势图, backtest, 回测, equity curve]
---

## Core rule

When creating charts/plots in Python execution:
1. Figures are **automatically captured** as PNG images after execution
2. Set a descriptive **title** using `plt.title()` or `fig.suptitle()` - this helps users understand the chart
3. Always use `plt.tight_layout()` before finishing to prevent label cutoff
4. **DO NOT** call `plt.show()` or `plt.savefig()` - figures are captured automatically

## When to create charts

- **Backtests**: Always plot equity curve + drawdown for backtest results
- **Price analysis**: Trends with indicators (MA, Bollinger, MACD)
- **Comparisons**: Multi-stock normalized performance overlay
- **Distributions**: Return histograms, volatility analysis
- **Correlations**: Heatmaps for correlation matrices

## Chinese font support

For Chinese labels/titles, configure matplotlib at the start:

```python
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display
```

## Basic price chart

```python
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

df = store.daily(ts_code, start_date="20240101")
df = df.sort_values("trade_date")

plt.figure(figsize=(12, 6))
plt.plot(df["trade_date"], df["close"], label="收盘价", color="#1f77b4")
plt.title(f"{ts_code} 股价走势")
plt.xlabel("日期")
plt.ylabel("价格 (元)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
# Figure is automatically captured - no need to save/show
```

## Price with moving averages

```python
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

df = store.daily(ts_code, start_date="20240101")
df = df.sort_values("trade_date")
df["ma5"] = df["close"].rolling(5).mean()
df["ma20"] = df["close"].rolling(20).mean()
df["ma60"] = df["close"].rolling(60).mean()

plt.figure(figsize=(12, 6))
plt.plot(df["trade_date"], df["close"], label="收盘价", color="#1f77b4", linewidth=1.5)
plt.plot(df["trade_date"], df["ma5"], label="MA5", color="#ff7f0e", linewidth=1)
plt.plot(df["trade_date"], df["ma20"], label="MA20", color="#2ca02c", linewidth=1)
plt.plot(df["trade_date"], df["ma60"], label="MA60", color="#d62728", linewidth=1)
plt.title(f"{ts_code} 股价与均线")
plt.xlabel("日期")
plt.ylabel("价格 (元)")
plt.xticks(rotation=45)
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
```

## Price + Volume subplots

```python
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

df = store.daily(ts_code, start_date="20240101")
df = df.sort_values("trade_date")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                               gridspec_kw={"height_ratios": [3, 1]})

# Price chart
ax1.plot(df["trade_date"], df["close"], color="#1f77b4", linewidth=1.5)
ax1.set_ylabel("价格 (元)")
ax1.set_title(f"{ts_code} 股价与成交量")
ax1.grid(True, alpha=0.3)

# Volume chart
colors = ["#2ca02c" if c >= 0 else "#d62728" for c in df["pct_chg"].fillna(0)]
ax2.bar(df["trade_date"], df["vol"], color=colors, alpha=0.7)
ax2.set_xlabel("日期")
ax2.set_ylabel("成交量 (手)")
ax2.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
```

## Multiple stocks comparison

```python
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

stocks = {"600519.SH": "贵州茅台", "000858.SZ": "五粮液", "000568.SZ": "泸州老窖"}
fig, ax = plt.subplots(figsize=(12, 6))

for ts_code, name in stocks.items():
    df = store.daily(ts_code, start_date="20240101")
    df = df.sort_values("trade_date")
    # Normalize to 100 at start for comparison
    df["normalized"] = df["close"] / df["close"].iloc[0] * 100
    ax.plot(df["trade_date"], df["normalized"], label=name, linewidth=1.5)

ax.set_title("白酒龙头股 归一化走势对比")
ax.set_xlabel("日期")
ax.set_ylabel("归一化价格 (起点=100)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
```

## Candlestick chart (using mplfinance-style approach)

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

df = store.daily(ts_code, start_date="20240101").sort_values("trade_date").tail(60)

fig, ax = plt.subplots(figsize=(14, 7))

for i, (_, row) in enumerate(df.iterrows()):
    color = "#d62728" if row["close"] < row["open"] else "#2ca02c"
    # Wick (high-low line)
    ax.plot([i, i], [row["low"], row["high"]], color=color, linewidth=1)
    # Body (open-close box)
    body_bottom = min(row["open"], row["close"])
    body_height = abs(row["close"] - row["open"])
    rect = mpatches.Rectangle((i - 0.3, body_bottom), 0.6, body_height, 
                                color=color, alpha=0.9)
    ax.add_patch(rect)

ax.set_title(f"{ts_code} K线图 (最近60日)")
ax.set_xlabel("交易日")
ax.set_ylabel("价格 (元)")
ax.set_xticks(range(0, len(df), 10))
ax.set_xticklabels(df["trade_date"].iloc[::10], rotation=45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
```

## Backtest equity curve + drawdown (IMPORTANT for backtests!)

Always generate this chart after running a backtest:

```python
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# Assuming df has 'equity' (cumulative returns) and we compute drawdown
df["peak"] = df["equity"].cummax()
df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"] * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})

# Equity curve
ax1.plot(df["trade_date"], df["equity"], color="#1f77b4", linewidth=1.5, label="策略净值")
ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
ax1.set_ylabel("净值")
ax1.set_title("MA双均线策略回测 - 600519.SH")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# Drawdown
ax2.fill_between(df["trade_date"], df["drawdown"], 0, color="#d62728", alpha=0.5)
ax2.set_ylabel("回撤 (%)")
ax2.set_xlabel("日期")
ax2.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
```

## Return distribution histogram

```python
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

returns = df["pct_chg"].dropna()

plt.figure(figsize=(10, 6))
plt.hist(returns, bins=50, edgecolor="black", alpha=0.7)
plt.axvline(x=returns.mean(), color="red", linestyle="--", label=f"均值: {returns.mean():.2f}%")
plt.axvline(x=0, color="gray", linestyle="-", alpha=0.5)
plt.title("日收益率分布 - 600519.SH")
plt.xlabel("日收益率 (%)")
plt.ylabel("频数")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
```

## Correlation heatmap

```python
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# Assuming corr_matrix is a pandas DataFrame with correlation values
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix.values, cmap="RdYlGn", vmin=-1, vmax=1)

ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.index)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
ax.set_yticklabels(corr_matrix.index)

# Add correlation values as text
for i in range(len(corr_matrix.index)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                       ha="center", va="center", fontsize=9)

plt.colorbar(im, label="相关系数")
plt.title("白酒股票收益率相关性")
plt.tight_layout()
```

## Style tips

1. **Figure size**: Use `figsize=(12, 6)` for single charts, `(12, 8)` for multi-panel
2. **Colors**: Use colorblind-friendly palettes; green/red for up/down
3. **Grid**: Light grid (`alpha=0.3`) improves readability
4. **Labels**: Always label axes and add legends
5. **Title**: Descriptive title helps users understand context
6. **Rotation**: Rotate date labels 45° to prevent overlap
