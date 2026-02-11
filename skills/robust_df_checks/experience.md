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
### ⚠️ Date type normalization (CRITICAL)
Date columns (`trade_date`, `end_date`, etc.) are often strings. **Comparing them with int literals causes TypeError.**

```python
# WRONG - raises TypeError: '<' not supported between 'str' and 'int'
df[df["trade_date"] >= 20240101]  # ❌

# RIGHT - normalize to int first
df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").astype(int)
df[df["trade_date"] >= 20240101]  # ✅

# OR use string comparison (works if format is consistent YYYYMMDD)
df[df["trade_date"].astype(str) >= "20240101"]  # ✅
```

### Safe iloc access (avoid IndexError on empty df)

```python
# WRONG - crashes if df is empty
first_row = df.iloc[0]  # ❌ IndexError if df.empty

# RIGHT - check first
if not df.empty:
    first_row = df.iloc[0]
else:
    raise ValueError("DataFrame is empty, cannot access first row")
```
## Common bugs to avoid
- Using `.iloc[0]` on empty df (crash).
- Assuming a field exists across all datasets.
- Rolling indicators on unsorted dates.
- **Comparing string date columns with int literals** (always normalize types first).

