# Routine Prompt: Trace Failure Triage → Code Fixes → Skills Upgrade

Use this prompt to run a repeatable workflow on `traces/*.jsonl` conversation logs.

## Refined Request Prompt (copy/paste)

You are working in this repo: `a-share-agent`.

Goal: Analyze the agent conversation logs in `traces/` and improve reliability.

Requirements:
1) Parse `traces/*.jsonl` and **filter failed Python executions** (tool calls to Python execution and their error output).
2) Cluster the failures and report **3–5 main root causes**, each with:
   - a short title
   - what it looks like (typical exception messages)
   - likely underlying cause
   - a minimal, robust fix strategy
3) Implement fixes in the codebase so they actually run:
   - Prioritize fixes that reduce recurrence across many traces
   - Avoid changing unrelated behavior
4) Decide which **skills should be upgraded** based on those root causes; upgrade only the most relevant ones.
5) Validate:
   - Add/adjust targeted tests when reasonable
   - Run `pytest` and ensure it passes

Constraints:
- Fix root causes, not superficial patches.
- Keep changes minimal and consistent with existing patterns.
- Don’t invent new UX or features beyond what’s required.
- If you are unsure, prefer safer defaults and add helpful hints/errors.

Deliverables:
- A short report: “Top root causes (3–5) → fixes applied → skills upgraded → tests run”.
- Links to the files changed.

## What counts as a “failed Python execution”
- Any `tool_execute_python` (or equivalent python sandbox tool) call with `success=false`, OR
- Exceptions in the tool’s `error` payload (e.g., `TypeError`, `KeyError`, `BinderException`, `ModuleNotFoundError`, etc.).

## Output Template (recommended)

**Root Causes**
- (1) <Title>
  - Signature:
  - Cause:
  - Fix:
  - Affected traces:

**Code Changes**
- <file>: <what changed>

**Skills Upgraded**
- <skill>: <what changed>

**Validation**
- `pytest`:

## Common high-value fix targets (examples)
- Store API misuse (`offset`, unsupported filters)
- DuckDB/Parquet binder errors (missing `ts_code` in dataset)
- Date dtype mismatches (`trade_date` string vs int comparisons)
- Empty-frame and missing-column guards for pandas workflows
- Alignment-first correlation/covariance patterns
- Optional dependencies (`scipy`, `matplotlib`) errors → graceful fallback or skill guidance
