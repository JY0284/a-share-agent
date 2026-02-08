````markdown
---
name: statistical_analysis
description: Statistical analysis for stock data using statsmodels - regression, time series (ARIMA), stationarity tests (ADF), cointegration, factor models, and hypothesis testing.
tags: [stats, statsmodels, regression, OLS, ARIMA, ADF, cointegration, 统计分析, 回归, 时间序列, 协整, alpha, beta, factor]
---

## Core rule
Use `statsmodels` for rigorous statistical analysis. Always check assumptions (stationarity, normality, heteroscedasticity) before interpreting results.

## Import convention

```python
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
```

## Recommended patterns

### 1) OLS Regression - Alpha/Beta (CAPM)

Compute stock's alpha and beta relative to a benchmark (e.g., CSI 300).

```python
# Load stock and benchmark returns
df_stock = store.daily_adj(ts_code, how="qfq", start_date="20230101").sort_values("trade_date")
df_bench = store.index_daily("000300.SH", start_date="20230101").sort_values("trade_date")

# Merge on trade_date
df_stock["ret"] = df_stock["close"].pct_change()
df_bench["ret_mkt"] = df_bench["close"].pct_change()
merged = pd.merge(
    df_stock[["trade_date", "ret"]],
    df_bench[["trade_date", "ret_mkt"]],
    on="trade_date",
    how="inner"
).dropna()

# OLS: ret = alpha + beta * ret_mkt + epsilon
X = sm.add_constant(merged["ret_mkt"])  # add intercept
y = merged["ret"]
model = OLS(y, X).fit()

alpha = float(model.params["const"])
beta = float(model.params["ret_mkt"])
r_squared = float(model.rsquared)
alpha_ann = alpha * 252  # annualized alpha

result = {
    "ts_code": ts_code,
    "alpha_daily": round(alpha, 6),
    "alpha_ann(%)": round(alpha_ann * 100, 2),
    "beta": round(beta, 3),
    "r_squared": round(r_squared, 3),
    "n_obs": int(model.nobs)
}
print(model.summary())
```

### 2) ADF Test - Check Stationarity

Test if a price/return series is stationary (required for many time series models).

```python
df = store.daily_adj(ts_code, how="qfq", start_date="20230101").sort_values("trade_date")
price = df["close"].dropna()

# ADF test on price (usually non-stationary)
adf_price = adfuller(price, maxlag=None, autolag="AIC")
# ADF test on returns (usually stationary)
returns = price.pct_change().dropna()
adf_ret = adfuller(returns, maxlag=None, autolag="AIC")

result = {
    "ts_code": ts_code,
    "price_adf_stat": round(adf_price[0], 4),
    "price_p_value": round(adf_price[1], 4),
    "price_stationary": adf_price[1] < 0.05,
    "return_adf_stat": round(adf_ret[0], 4),
    "return_p_value": round(adf_ret[1], 4),
    "return_stationary": adf_ret[1] < 0.05,
}
# p < 0.05 → reject null (series is stationary)
```

### 3) Cointegration Test - Pairs Trading

Test if two stocks are cointegrated (mean-reverting spread).

```python
ts_code1, ts_code2 = "600519.SH", "000858.SZ"  # e.g., Moutai vs Wuliangye

df1 = store.daily_adj(ts_code1, how="qfq", start_date="20230101").sort_values("trade_date")
df2 = store.daily_adj(ts_code2, how="qfq", start_date="20230101").sort_values("trade_date")

merged = pd.merge(
    df1[["trade_date", "close"]].rename(columns={"close": "p1"}),
    df2[["trade_date", "close"]].rename(columns={"close": "p2"}),
    on="trade_date", how="inner"
).dropna()

# Engle-Granger cointegration test
coint_stat, p_value, crit_values = coint(merged["p1"], merged["p2"])

result = {
    "pair": f"{ts_code1} vs {ts_code2}",
    "coint_stat": round(coint_stat, 4),
    "p_value": round(p_value, 4),
    "cointegrated": p_value < 0.05,
    "crit_1%": round(crit_values[0], 4),
    "crit_5%": round(crit_values[1], 4),
    "n_obs": len(merged)
}
# p < 0.05 → cointegrated, spread is mean-reverting
```

### 4) ARIMA Model - Time Series Forecasting

Fit ARIMA model for return forecasting (use with caution for actual trading).

```python
df = store.daily_adj(ts_code, how="qfq", start_date="20230101").sort_values("trade_date")
returns = df["close"].pct_change().dropna()

# Fit ARIMA(1,0,1) on returns (stationary series, d=0)
model = ARIMA(returns, order=(1, 0, 1))
fitted = model.fit()

# Forecast next 5 days
forecast = fitted.forecast(steps=5)

result = {
    "ts_code": ts_code,
    "aic": round(fitted.aic, 2),
    "bic": round(fitted.bic, 2),
    "forecast_5d": forecast.tolist()
}
print(fitted.summary())
```

### 5) Rolling Beta / Time-Varying Beta

Compute rolling beta to see how sensitivity to market changes over time.

```python
df_stock = store.daily_adj(ts_code, how="qfq", start_date="20220101").sort_values("trade_date")
df_bench = store.index_daily("000300.SH", start_date="20220101").sort_values("trade_date")

df_stock["ret"] = df_stock["close"].pct_change()
df_bench["ret_mkt"] = df_bench["close"].pct_change()
merged = pd.merge(
    df_stock[["trade_date", "ret"]],
    df_bench[["trade_date", "ret_mkt"]],
    on="trade_date", how="inner"
).dropna().reset_index(drop=True)

window = 60  # 60-day rolling window
betas = []
for i in range(window, len(merged)):
    sub = merged.iloc[i-window:i]
    X = sm.add_constant(sub["ret_mkt"])
    y = sub["ret"]
    beta = OLS(y, X).fit().params["ret_mkt"]
    betas.append({"trade_date": merged.iloc[i]["trade_date"], "rolling_beta": beta})

result = pd.DataFrame(betas).tail(20)
```

### 6) Factor Regression - Multi-Factor Model

Regress stock returns on multiple factors (e.g., market, size, value).

```python
# Assume you have factor returns: mkt, smb (size), hml (value)
# This is a simplified example - real factor data needs proper construction

df_stock = store.daily_adj(ts_code, how="qfq", start_date="20230101").sort_values("trade_date")
df_stock["ret"] = df_stock["close"].pct_change()

# Example: load market and simulate factors (replace with real factor data)
df_mkt = store.index_daily("000300.SH", start_date="20230101").sort_values("trade_date")
df_mkt["mkt"] = df_mkt["close"].pct_change()

merged = pd.merge(
    df_stock[["trade_date", "ret"]],
    df_mkt[["trade_date", "mkt"]],
    on="trade_date", how="inner"
).dropna()

# Multi-factor OLS (add more factors as available)
X = sm.add_constant(merged[["mkt"]])  # extend with ["mkt", "smb", "hml"]
y = merged["ret"]
model = OLS(y, X).fit()

result = {
    "ts_code": ts_code,
    "alpha": round(model.params["const"], 6),
    "beta_mkt": round(model.params["mkt"], 3),
    "r_squared": round(model.rsquared, 3),
    "adj_r_squared": round(model.rsquared_adj, 3)
}
print(model.summary())
```

### 7) Autocorrelation Analysis

Check for autocorrelation in returns (market efficiency test).

```python
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

df = store.daily_adj(ts_code, how="qfq", start_date="20230101").sort_values("trade_date")
returns = df["close"].pct_change().dropna()

# Durbin-Watson statistic (2 = no autocorrelation)
dw_stat = durbin_watson(returns)

# Ljung-Box test for autocorrelation at multiple lags
lb_test = acorr_ljungbox(returns, lags=[5, 10, 20], return_df=True)

result = {
    "ts_code": ts_code,
    "durbin_watson": round(dw_stat, 4),
    "lb_stat_lag5": round(lb_test.loc[5, "lb_stat"], 4),
    "lb_pval_lag5": round(lb_test.loc[5, "lb_pvalue"], 4),
    "autocorr_sig_lag5": lb_test.loc[5, "lb_pvalue"] < 0.05
}
```

### 8) Heteroscedasticity Test

Test for non-constant variance in regression residuals.

```python
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

# After fitting OLS model (from alpha/beta example)
# model = OLS(y, X).fit()

bp_stat, bp_pval, _, _ = het_breuschpagan(model.resid, model.model.exog)
white_stat, white_pval, _, _ = het_white(model.resid, model.model.exog)

result = {
    "breusch_pagan_stat": round(bp_stat, 4),
    "breusch_pagan_pval": round(bp_pval, 4),
    "white_stat": round(white_stat, 4),
    "white_pval": round(white_pval, 4),
    "heteroscedastic": bp_pval < 0.05 or white_pval < 0.05
}
# If heteroscedastic, use robust standard errors: OLS(...).fit(cov_type='HC3')
```

## Common bugs to avoid
- Running ARIMA on non-stationary data (difference first or use d>0).
- Forgetting to add constant in OLS (use `sm.add_constant()`).
- Using price levels for regression (use returns instead).
- Interpreting daily alpha as annual (multiply by 252).
- Not checking p-values before trusting coefficients.
- Overfitting ARIMA with high p,d,q (use AIC/BIC for model selection).

## Interpretation guide
- **Beta > 1**: More volatile than market; **Beta < 1**: Less volatile
- **Alpha > 0**: Outperformance vs benchmark (risk-adjusted)
- **ADF p < 0.05**: Series is stationary
- **Cointegration p < 0.05**: Pair is cointegrated (mean-reverting)
- **R² close to 1**: Model explains most variance

## See also
- `risk_metrics`: volatility and drawdown calculations
- `rolling_indicators`: technical indicators (MA, RSI, MACD)
- `adj_prices_and_returns`: return computation basics
````
