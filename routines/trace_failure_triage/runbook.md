# Runbook: Trace Failure Triage → Fixes → Skills Upgrade

## Quick start

1) Scan traces for Python failures
- Look for `tool_execute_python` blocks with `success=false`.
- Extract the exception type + message.

2) Cluster into 3–5 buckets
- Group by root cause, not by exact message.
- Prefer the smallest number of buckets that explains most failures.

3) Apply the smallest robust fixes
- Prefer centralized fixes (sandbox/tooling) over per-strategy patches.
- Add actionable hints when the agent makes a common mistake.

4) Upgrade skills
- Update only the skills that directly prevent the top buckets.
- Add a “Common pitfalls (from traces)” section when helpful.

5) Validate
- Add targeted unit tests where feasible.
- Run `pytest`.

## Repo-specific tips

- The python sandbox is the best place for cross-cutting improvements.
- Skills should teach:
  - correct store API usage (avoid unsupported args)
  - date dtype normalization for comparisons/slicing
  - empty-data guards and required column checks
  - alignment-first correlation/covariance

## Done definition
- Report produced (3–5 root causes).
- Fixes implemented and tests passing.
- Skills upgraded in a trace-driven, minimal way.
