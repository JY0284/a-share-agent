---
name: robust_df_checks
description: Defensive checks for empty DataFrames, missing columns, and date ranges before calculating.
tags: [pandas, robustness, empty, columns]
---

## Core rule
Before any calculation, confirm:
1) DataFrame not empty
2) Required columns exist
3) Data is sorted correctly for the algorithm

## Recommended patterns

### Empty checks

```python
df = store.daily(ts_code, start_date="20240101")
if df.empty:
    raise ValueError(f\"No daily data for {ts_code} in given range\")
```

### Column checks

```python
need = [\"trade_date\", \"close\"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise KeyError(f\"Missing columns: {missing}; got: {df.columns.tolist()}\")
```

### Safe sorting

```python
df = df.sort_values(\"trade_date\")
```

## Common bugs to avoid
- Using `.iloc[0]` on empty df (crash).
- Assuming a field exists across all datasets.
- Rolling indicators on unsorted dates.

