---
name: trace_failure_triage
description: Standard workflow to analyze traces/*.jsonl for failed Python executions, cluster 3–5 root causes, implement fixes, and upgrade skills with tests.
tags: [traces, debugging, reliability, agent, skills, python, pandas, triage, maintenance]
---

## When to use
Use this skill whenever the agent produces repeated `tool_execute_python` failures, or after a run where `traces/*.jsonl` contains many errors.

## Goal
Turn trace logs into:
1) **3–5 root-cause buckets**
2) **a clear summary of user needs** (what the user asked the agent to do)
2) **code fixes that actually run** (prefer centralized fixes)
3) **skill upgrades** that prevent recurrence
4) **tests passing** (`pytest`)

## Definition: “failed Python execution”
- Any `tool_execute_python` call with `success=false`, OR
- Any tool output with a Python exception in its `error` payload.

## Routine (standard steps)
1) Extract user messages (user needs)
- For each trace, capture:
    - the **latest user request** (and optionally the full list of user turns)
    - any hard constraints (symbols, date range, hfq/qfq, ETF vs stock, output format)

2) Scan `traces/*.jsonl` and extract failed python executions
- Capture: trace file, exception type/message, the code snippet (if present), and the **nearest preceding user request**.

3) Cluster into **3–5** buckets (root causes)
Common buckets in this repo:
- Store API misuse (e.g. `offset`, unsupported keyword args)
- DuckDB/Parquet binder errors (missing columns like `ts_code`)
- Date dtype mismatch (`trade_date` string vs int comparisons)
- Backtest price-type misuse (qfq vs hfq for stocks)
- ETF data access misuse (trying to use stock adjusted-price APIs on ETFs)
- Empty data / missing columns (`KeyError`, `NameError` from branchy code)
- Misaligned arrays / correlation/covariance built from unaligned series
- Optional deps missing (`scipy`, `matplotlib`) → degrade gracefully / adjust skills
- LLM code-gen mistakes (e.g. malformed f-string format specifiers)

4) Summarize user needs (3–5 intents) and map to buckets
- Produce a short “**User intents**” list and for each intent:
    - what the user asked for
    - which failure bucket(s) prevented it
    - what a “correct” output would have looked like

5) Fix the codebase (minimal, high-leverage)
- Prefer fixes in the Python sandbox/tooling layer (one fix helps many traces).
- Add actionable error hints rather than silent coercions when behavior could surprise.

6) Upgrade skills
- Upgrade only the skills that directly prevent the top buckets.
- Add a short “Common pitfalls (from traces)” section.

7) Validate
- Add 1–3 targeted tests for new behavior.
- Run `pytest` and confirm all pass.

## Refined request prompt (for any agent)
Copy/paste into an agent session:

- Analyze `traces/*.jsonl` and extract user messages to understand user needs.
- Analyze `traces/*.jsonl` and filter failed python executions.
- Summarize **3–5 main failure reasons** with signatures + fix strategies.
- Summarize **3–5 user intents** and map them to the failure buckets.
- Implement fixes (prioritize cross-cutting reliability).
- Upgrade only the most relevant skills.
- Add/adjust targeted tests and run `pytest`.

Extra triage reminders (based on recent traces):
- Date compare errors often include `TypeError: '<' not supported between instances of 'str' and 'int'` or Arrow `Function 'less' has no kernel ... (large_string, int64)`.
- Stock backtests: require `store.daily_adj(ts_code, how="hfq")` (后复权) by default.
- ETFs: **no adj_factor / no daily_adj**; use `store.etf_daily(ts_code, ...)` for price bars and `store.fund_nav(ts_code, ...)` for NAV when needed.
- If you see `ValueError: No daily_adj data for 510300.SH/513100.SH`, check if the agent mistakenly used stock adjusted-price APIs for an ETF.

(Full template also lives in `routines/trace_failure_triage/prompt.md`.)

## Tooling snippet: scan traces for failures + user messages (tested)
Run inside `tool_execute_python` (or any Python runner) from repo root.

```python
import json
from pathlib import Path

TRACE_DIR = Path("traces")

# If your traces live elsewhere (e.g. a sibling folder), point TRACE_DIR at that path.

# Heuristics: trace schemas differ across frameworks. Keep this intentionally flexible.
def is_failed_python_tool_event(obj: dict) -> bool:
    # Flexible: trace schemas differ. This is intentionally heuristic.
    text = json.dumps(obj, ensure_ascii=False)
    if "tool_execute_python" not in text:
        return False
    if '"success": false' in text or "'success': False" in text:
        return True
    if "Traceback" in text and "error" in text:
        return True
    return False


def extract_user_message(obj: dict) -> str | None:
    """Best-effort extraction of a user turn from a trace event."""
    # Common patterns seen in various trace formats:
    # - {"role": "user", "content": "..."}
    # - {"type": "human", "content": "..."}
    # - nested messages arrays
    try:
        role = obj.get("role")
        if role == "user" and isinstance(obj.get("content"), str):
            return obj["content"].strip()
        if obj.get("type") in {"human", "user"} and isinstance(obj.get("content"), str):
            return obj["content"].strip()
    except Exception:
        pass

    msgs = obj.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict):
                txt = extract_user_message(m)
                if txt:
                    return txt
    return None


records = []
max_files = 200
max_lines_per_file = 5000

paths = sorted(TRACE_DIR.glob("*.jsonl"))[:max_files]
for p in paths:
    try:
        last_user = None
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= max_lines_per_file:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                umsg = extract_user_message(obj)
                if umsg:
                    last_user = umsg

                if is_failed_python_tool_event(obj):
                    records.append(
                        {
                            "file": p.name,
                            "line": i + 1,
                            "last_user_message": last_user,
                        }
                    )
                    break
    except Exception:
        continue

result = {
    "n_trace_files_checked": len(paths),
    "n_files_with_fail_event": len(records),
    "sample": records[:10],
}
result
```

## Expected report format
- **User intents (3–5)**: intent → constraints → what success looks like
- **Root Causes (3–5)**: title → signature → cause → fix
- **Code changes**: file → what changed
- **Skills upgraded**: skill → what changed
- **Validation**: `pytest` result
