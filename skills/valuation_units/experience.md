---
name: valuation_units
description: Unit conversions and safe selection for daily_basic fields (PE/PB/市值/换手).
tags: [daily_basic, PE, PB, 市值, 换手, units]
---

## Core rule
Always state units. Convert to human-friendly units when presenting.

## Common fields (typical)
- `pe_ttm`, `pb`
- `total_mv`, `circ_mv` (often in 万元)
- `turnover_rate` (%)

## Recommended patterns

### Latest valuation row

```python
basic = store.daily_basic(ts_code).sort_values("trade_date", ascending=False).head(1)
row = basic.iloc[0]
```

### Market cap conversions
If `total_mv` is in 万元, then:

```python
total_mv_yi = row["total_mv"] / 10000  # 亿元
circ_mv_yi = row["circ_mv"] / 10000
```

### Present a compact table

```python
result = pd.DataFrame([{
  "ts_code": ts_code,
  "trade_date": row["trade_date"],
  "pe_ttm": float(row["pe_ttm"]),
  "pb": float(row["pb"]),
  "total_mv(亿)": round(total_mv_yi, 1),
  "circ_mv(亿)": round(circ_mv_yi, 1),
  "turnover_rate(%)": float(row.get("turnover_rate", np.nan)),
}])
```

## Common bugs to avoid
- Mixing “万/亿/元” in one answer.
- Assuming total_mv is already in 亿元.
- Forgetting to label percent fields.

