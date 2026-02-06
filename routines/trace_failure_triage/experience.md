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
2) **code fixes that actually run** (prefer centralized fixes)
3) **skill upgrades** that prevent recurrence
4) **tests passing** (`pytest`)

## Definition: “failed Python execution”
- Any `tool_execute_python` call with `success=false`, OR
- Any tool output with a Python exception in its `error` payload.

## Routine (standard steps)
1) Scan `traces/*.jsonl` and extract failed python executions
- Capture: trace file, exception type/message, and the code snippet (if present).

2) Cluster into **3–5** buckets (root causes)
Common buckets in this repo:
- Store API misuse (e.g. `offset`, unsupported keyword args)
- DuckDB/Parquet binder errors (missing columns like `ts_code`)
- Date dtype mismatch (`trade_date` string vs int comparisons)
- Empty data / missing columns (`KeyError`, `NameError` from branchy code)
- Misaligned arrays / correlation/covariance built from unaligned series
- Optional deps missing (`scipy`, `matplotlib`) → degrade gracefully / adjust skills

3) Fix the codebase (minimal, high-leverage)
- Prefer fixes in the Python sandbox/tooling layer (one fix helps many traces).
- Add actionable error hints rather than silent coercions when behavior could surprise.

4) Upgrade skills
- Upgrade only the skills that directly prevent the top buckets.
- Add a short “Common pitfalls (from traces)” section.

5) Validate
- Add 1–3 targeted tests for new behavior.
- Run `pytest` and confirm all pass.

## Refined request prompt (for any agent)
Copy/paste into an agent session:

- Analyze `traces/*.jsonl` and filter failed python executions.
- Summarize **3–5 main failure reasons** with signatures + fix strategies.
- Implement fixes (prioritize cross-cutting reliability).
- Upgrade only the most relevant skills.
- Add/adjust targeted tests and run `pytest`.

(Full template also lives in `routines/trace_failure_triage/prompt.md`.)

## Tooling snippet: scan traces for failures (tested)
Run inside `tool_execute_python` (or any Python runner) from repo root.

```python
import json
from pathlib import Path

TRACE_DIR = Path("traces")

# Heuristics: these keys show up in many trace tool results
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


failures = []
max_files = 200
max_lines_per_file = 5000

paths = sorted(TRACE_DIR.glob("*.jsonl"))[:max_files]
for p in paths:
    try:
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
                if is_failed_python_tool_event(obj):
                    failures.append({"file": p.name, "line": i + 1})
                    break
    except Exception:
        continue

result = {
    "n_trace_files_checked": len(paths),
    "n_files_with_fail_event": len(failures),
    "sample": failures[:10],
}
result
```

## Expected report format
- **Root Causes (3–5)**: title → signature → cause → fix
- **Code changes**: file → what changed
- **Skills upgraded**: skill → what changed
- **Validation**: `pytest` result
