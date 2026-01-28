---
name: finance_statements_metrics
description: Pull quarterly/annual statements (income/balancesheet/cashflow/fina_indicator) and compute a small set of robust ratios.
tags: [finance, income, balancesheet, cashflow, fina_indicator, 财务, 利润表, 资产负债表, 现金流量表, ratios, 同比]
---

## Core rule
Finance datasets are **report-period** data keyed by `end_date` (e.g., 20240930, 20241231), not trading days.

Use `start_period` / `end_period` for filtering by report periods, then compute metrics after sorting by `end_date`.

## Recommended patterns

### Load recent income statements and compute margins (best-effort columns)

```python
inc = store.income("600519.SH", start_period="20200101")
inc = inc.sort_values("end_date")

# Column names depend on tushare schema; check what's available
cols = set(inc.columns)
need = []
for c in ["end_date", "revenue", "total_revenue", "n_income", "n_income_attr_p", "operate_profit"]:
    if c in cols:
        need.append(c)

view = inc[need].tail(12).copy()

# Example: net margin using the best available columns
rev_col = "revenue" if "revenue" in view.columns else ("total_revenue" if "total_revenue" in view.columns else None)
ni_col = "n_income_attr_p" if "n_income_attr_p" in view.columns else ("n_income" if "n_income" in view.columns else None)
if rev_col and ni_col:
    view["net_margin"] = view[ni_col] / view[rev_col]

view.tail(8)
```

### Load balance sheet and compute leverage ratios (best-effort columns)

```python
bs = store.balancesheet("600519.SH", start_period="20200101").sort_values("end_date")
cols = set(bs.columns)

need = [c for c in ["end_date", "total_assets", "total_hldr_eqy_exc_min_int", "total_liab"] if c in cols]
view = bs[need].tail(12).copy()

if "total_liab" in view.columns and "total_assets" in view.columns:
    view["liab_to_assets"] = view["total_liab"] / view["total_assets"]

view.tail(8)
```

### Load cashflow and compute CFO-to-net-income (best-effort columns)

```python
cf = store.cashflow("600519.SH", start_period="20200101").sort_values("end_date")
inc = store.income("600519.SH", start_period="20200101").sort_values("end_date")

cf_cols = set(cf.columns)
inc_cols = set(inc.columns)

# Common cashflow columns in tushare: n_cashflow_act (经营活动产生的现金流量净额)
cfo_col = "n_cashflow_act" if "n_cashflow_act" in cf_cols else None
ni_col = "n_income_attr_p" if "n_income_attr_p" in inc_cols else ("n_income" if "n_income" in inc_cols else None)

if cfo_col and ni_col:
    merged = cf[["end_date", cfo_col]].merge(inc[["end_date", ni_col]], on="end_date", how="inner")
    merged["cfo_to_net_income"] = merged[cfo_col] / merged[ni_col]
    merged.tail(8)
```

## Common bugs to avoid
- Treating `end_date` like trading dates.
- Mixing quarterly and annual rows without filtering (check `report_type` if present).
- Hard-coding finance column names without checking if they exist in your local store.

