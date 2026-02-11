# Routine Prompt: Trace Failure Triage → Code Fixes → Skills Upgrade

Use this prompt to run a repeatable workflow on `traces/*.jsonl` conversation logs.

## Refined Request Prompt (copy/paste)

You are working in this repo: `a-share-agent`.

Goal: Analyze the agent conversation logs in `traces/` and improve reliability.

Requirements:
1) Parse `traces/*.jsonl` and extract **user messages** to understand user needs.
   - Summarize **3–5 user intents** (what the user wanted), and capture key constraints (symbols, dates, ETF vs stock, hfq/qfq, output format).
2) Parse `traces/*.jsonl` and **filter failed Python executions** (tool calls to Python execution and their error output).
   - Keep the nearest preceding user request for each failure (so errors stay tied to intent).
3) Cluster the failures and report **3–5 main root causes**, each with:
   - a short title
   - what it looks like (typical exception messages)
   - likely underlying cause
   - a minimal, robust fix strategy
4) Implement fixes in the codebase so they actually run:
   - Prioritize fixes that reduce recurrence across many traces
   - Avoid changing unrelated behavior
5) Decide which **skills should be upgraded** based on those root causes; upgrade only the most relevant ones.
6) Validate:
   - Add/adjust targeted tests when reasonable
   - Run `pytest` and ensure it passes

Constraints:
- Fix root causes, not superficial patches.
- Keep changes minimal and consistent with existing patterns.
- Don’t invent new UX or features beyond what’s required.
- If you are unsure, prefer safer defaults and add helpful hints/errors.

Deliverables:
- A short report: “User intents (3–5) → root causes (3–5) → fixes applied → skills upgraded → tests run”.
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

**User Intents**
- (1) <Intent>
   - Constraints:
   - Blocked by:
   - What success looks like:

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
- Backtests using the wrong price type (stocks should default to 后复权 `hfq`)
- ETF data misuse (ETFs have no adj_factor / no daily_adj; use `store.etf_daily` + optional `store.fund_nav`)
- Empty-frame and missing-column guards for pandas workflows
- Alignment-first correlation/covariance patterns
- Optional dependencies (`scipy`, `matplotlib`) errors → graceful fallback or skill guidance
- LLM code-gen mistakes that can be skill-guided (e.g. invalid f-string format specifiers)

## Repo-specific guidance (important)
- For **stock** backtests / return analysis, prefer `store.daily_adj(ts_code, how="hfq")`.
- For **ETF** analysis, do **not** use `daily_adj`; use `store.etf_daily(ts_code, ...)` for price bars.
- When filtering by dates in pandas, normalize `trade_date` types first (avoid `str` vs `int` compare errors).
