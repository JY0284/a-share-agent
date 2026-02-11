# Runbook: Trace Failure Triage → Fixes → Skills Upgrade

## Quick start

1) Extract user needs from traces
- Skim each trace’s user turns to understand what output the user wanted.
- Capture constraints (symbols, date ranges, ETF vs stock, hfq/qfq, expected output format).

2) Scan traces for Python failures
- Look for `tool_execute_python` blocks with `success=false`.
- Extract the exception type + message.
- Keep the nearest preceding user request so failures stay grounded in intent.

3) Cluster into 3–5 buckets
- Group by root cause, not by exact message.
- Prefer the smallest number of buckets that explains most failures.

4) Map buckets to user intents
- For each user intent, note which bucket(s) blocked success and what “correct output” means.

5) Apply the smallest robust fixes
- Prefer centralized fixes (sandbox/tooling) over per-strategy patches.
- Add actionable hints when the agent makes a common mistake.

6) Upgrade skills
- Update only the skills that directly prevent the top buckets.
- Add a “Common pitfalls (from traces)” section when helpful.

7) Validate
- Add targeted unit tests where feasible.
- Run `pytest`.

## Repo-specific tips

- The python sandbox is the best place for cross-cutting improvements.
- Skills should teach:
  - correct store API usage (avoid unsupported args)
  - date dtype normalization for comparisons/slicing
  - stocks backtests: use 后复权 (hfq) prices via `store.daily_adj(ts_code, how="hfq")`
  - ETFs: no adj_factor / no daily_adj; use etf_daily + (optional) fund_nav
  - empty-data guards and required column checks
  - alignment-first correlation/covariance

## Trace-driven patterns (2026-02)

- Date compare failures show up as:
  - `TypeError: '<' not supported between instances of 'str' and 'int'`
  - `TypeError: Invalid comparison between dtype=str and int`
  - `pyarrow.lib.ArrowNotImplementedError: Function 'less' has no kernel ... (large_string, int64)`
  Fix is almost always: normalize `trade_date` to consistent type before filtering/slicing.

- ETF data-not-found failures often come from *using stock adjusted-price APIs on ETFs*:
  - `ValueError: No daily_adj data for 510300.SH/513100.SH`
  - `ValueError: 没有找到 513100.SH 的前复权数据` (agent tried to request 前复权 for ETF)
  - `ValueError: 没有找到 513100.SH 的日线数据` (no stored ETF bars for that code/range)
  Fix is: `store.etf_daily(ts_code, ...)` for price bars; optionally join `store.fund_nav(ts_code, ...)`.

- Backtest correctness: for stocks, require `store.daily_adj(ts_code, how='hfq')` (qfq is for charting, start point floats).

## Done definition
- Report produced (3–5 root causes).
- Fixes implemented and tests passing.
- Skills upgraded in a trace-driven, minimal way.
