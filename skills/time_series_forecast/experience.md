````markdown
---
name: time_series_forecast
description: Time series forecasting for stock data - ARIMA, GARCH volatility models, exponential smoothing, and trend decomposition using statsmodels.
tags: [forecast, ARIMA, GARCH, volatility, 预测, 波动率模型, time series, 时间序列, trend, seasonality]
---

## Core rule
Time series forecasting in finance has limited predictive power for prices. Focus on **volatility forecasting** (more reliable) and use price/return forecasts with extreme caution.

## Import convention

```python
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from arch import arch_model  # for GARCH (pip install arch)
import numpy as np
import pandas as pd
```

## Recommended patterns

### 1) ARIMA Model Selection with AIC/BIC

Find optimal ARIMA(p,d,q) parameters using information criteria.

```python
df = store.daily_adj(ts_code, how="qfq", start_date="20230101").sort_values("trade_date")
returns = df["close"].pct_change().dropna().values

# Grid search for best (p, q) with d=0 (returns are stationary)
best_aic = np.inf
best_order = None
best_model = None

for p in range(0, 4):
    for q in range(0, 4):
        try:
            model = ARIMA(returns, order=(p, 0, q))
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, 0, q)
                best_model = fitted
        except:
            continue

result = {
    "ts_code": ts_code,
    "best_order": best_order,
    "aic": round(best_aic, 2),
    "bic": round(best_model.bic, 2) if best_model else None
}
print(f"Best ARIMA{best_order} with AIC={best_aic:.2f}")
```

### 2) GARCH Volatility Forecasting

Model and forecast time-varying volatility (useful for risk management, options pricing).

```python
# Note: requires `arch` package: pip install arch
from arch import arch_model

df = store.daily_adj(ts_code, how="qfq", start_date="20220101").sort_values("trade_date")
returns = df["close"].pct_change().dropna() * 100  # Scale to percentage

# Fit GARCH(1,1) model
model = arch_model(returns, vol="Garch", p=1, q=1, mean="constant", dist="normal")
fitted = model.fit(disp="off")

# Forecast volatility for next 5 days
forecast = fitted.forecast(horizon=5)
vol_forecast = np.sqrt(forecast.variance.iloc[-1].values)  # Convert variance to std

result = {
    "ts_code": ts_code,
    "omega": round(fitted.params["omega"], 6),
    "alpha": round(fitted.params["alpha[1]"], 4),
    "beta": round(fitted.params["beta[1]"], 4),
    "persistence": round(fitted.params["alpha[1]"] + fitted.params["beta[1]"], 4),
    "vol_forecast_5d(%)": vol_forecast.round(4).tolist()
}
print(fitted.summary())
```

### 3) Exponential Smoothing (Holt-Winters)

Smooth and forecast series with trend (useful for fundamental metrics).

```python
df = store.daily_adj(ts_code, how="qfq", start_date="20230101").sort_values("trade_date")
price = df["close"].values

# Holt's method (trend, no seasonality for daily stock data)
model = ExponentialSmoothing(
    price,
    trend="add",
    seasonal=None,
    damped_trend=True
)
fitted = model.fit()

# Forecast next 20 days
forecast = fitted.forecast(20)

result = {
    "ts_code": ts_code,
    "smoothing_level": round(fitted.params["smoothing_level"], 4),
    "smoothing_trend": round(fitted.params["smoothing_trend"], 4),
    "forecast_20d": forecast.round(2).tolist()
}
```

### 4) Trend Decomposition

Decompose price/return series into trend, seasonal, and residual components.

```python
df = store.daily_adj(ts_code, how="qfq", start_date="20220101").sort_values("trade_date")
price = df.set_index("trade_date")["close"]
price.index = pd.to_datetime(price.index.astype(str))

# Decompose (period=20 for ~monthly cycle in daily data)
decomposition = seasonal_decompose(price, model="multiplicative", period=20)

# Get components
df_decomp = pd.DataFrame({
    "trend": decomposition.trend,
    "seasonal": decomposition.seasonal,
    "resid": decomposition.resid
}).dropna()

# Recent trend direction
recent_trend = df_decomp["trend"].iloc[-20:]
trend_direction = "up" if recent_trend.iloc[-1] > recent_trend.iloc[0] else "down"

result = {
    "ts_code": ts_code,
    "trend_direction": trend_direction,
    "trend_change_20d(%)": round((recent_trend.iloc[-1] / recent_trend.iloc[0] - 1) * 100, 2),
    "seasonal_range": round((df_decomp["seasonal"].max() - df_decomp["seasonal"].min()) * 100, 2)
}
```

### 5) ACF/PACF Analysis for Model Order Selection

Analyze autocorrelation to determine ARIMA order.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

df = store.daily_adj(ts_code, how="qfq", start_date="20230101").sort_values("trade_date")
returns = df["close"].pct_change().dropna()

# Compute ACF and PACF values
acf_vals = acf(returns, nlags=20)
pacf_vals = pacf(returns, nlags=20)

# Find significant lags (rough heuristic: |value| > 2/sqrt(n))
n = len(returns)
threshold = 2 / np.sqrt(n)
sig_acf = [i for i, v in enumerate(acf_vals[1:], 1) if abs(v) > threshold]
sig_pacf = [i for i, v in enumerate(pacf_vals[1:], 1) if abs(v) > threshold]

result = {
    "ts_code": ts_code,
    "n_obs": n,
    "sig_threshold": round(threshold, 4),
    "significant_acf_lags": sig_acf[:5],
    "significant_pacf_lags": sig_pacf[:5],
    "suggested_ar_order": len(sig_pacf[:3]),
    "suggested_ma_order": len(sig_acf[:3])
}

# Optional: plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(returns, lags=20, ax=axes[0], title="ACF")
plot_pacf(returns, lags=20, ax=axes[1], title="PACF")
plt.tight_layout()
plt.show()
```

### 6) Rolling Volatility Forecast Evaluation

Backtest GARCH volatility forecasts.

```python
from arch import arch_model

df = store.daily_adj(ts_code, how="qfq", start_date="20220101").sort_values("trade_date")
returns = (df["close"].pct_change().dropna() * 100).values

window = 252  # 1 year training window
forecasts = []
actuals = []

for i in range(window, len(returns) - 1):
    train = returns[i-window:i]
    model = arch_model(train, vol="Garch", p=1, q=1, mean="constant", rescale=False)
    fitted = model.fit(disp="off", show_warning=False)
    
    # 1-day ahead volatility forecast
    fc = fitted.forecast(horizon=1)
    vol_fc = np.sqrt(fc.variance.iloc[-1, 0])
    forecasts.append(vol_fc)
    
    # Actual realized volatility (absolute return as proxy)
    actuals.append(abs(returns[i]))

# Evaluation
forecasts = np.array(forecasts)
actuals = np.array(actuals)
mae = np.mean(np.abs(forecasts - actuals))
corr = np.corrcoef(forecasts, actuals)[0, 1]

result = {
    "ts_code": ts_code,
    "n_forecasts": len(forecasts),
    "mae": round(mae, 4),
    "correlation": round(corr, 4)
}
```

### 7) Regime Detection with Markov Switching

Detect bull/bear market regimes (advanced).

```python
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

df = store.daily_adj(ts_code, how="qfq", start_date="20200101").sort_values("trade_date")
returns = df["close"].pct_change().dropna() * 100

# Fit 2-regime Markov Switching model
model = MarkovRegression(returns, k_regimes=2, trend="c", switching_variance=True)
fitted = model.fit()

# Get regime probabilities
smoothed_probs = fitted.smoothed_marginal_probabilities

# Identify current regime
current_regime = smoothed_probs.iloc[-1].idxmax()
regime_names = {0: "low_vol", 1: "high_vol"}  # typically regime 0 has lower vol

result = {
    "ts_code": ts_code,
    "current_regime": regime_names.get(current_regime, f"regime_{current_regime}"),
    "regime_0_prob": round(smoothed_probs.iloc[-1, 0], 4),
    "regime_1_prob": round(smoothed_probs.iloc[-1, 1], 4),
    "regime_0_mean": round(fitted.params["const[0]"], 4),
    "regime_1_mean": round(fitted.params["const[1]"], 4)
}
print(fitted.summary())
```

## Common bugs to avoid
- Forecasting prices directly (non-stationary) instead of returns/differences.
- Using future data in backtest (lookahead bias).
- Not scaling returns for GARCH (use % not decimals).
- Ignoring model diagnostics (check residuals for autocorrelation).
- Over-parameterizing ARIMA (leads to overfitting).
- Expecting accurate price forecasts (volatility forecasts are more reliable).

## Model selection guide
| Goal | Model | Notes |
|------|-------|-------|
| Return forecasting | ARIMA | Limited accuracy for stocks |
| Volatility forecasting | GARCH(1,1) | Most common, reliable |
| Trend + level | Exponential Smoothing | Good for fundamentals |
| Regime detection | Markov Switching | Bull/bear identification |
| Decomposition | seasonal_decompose | Trend extraction |

## See also
- `statistical_analysis`: regression, hypothesis tests
- `risk_metrics`: historical volatility measures
- `rolling_indicators`: technical indicator calculations
````
