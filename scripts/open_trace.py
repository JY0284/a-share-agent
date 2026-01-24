#!/usr/bin/env python3
"""Open and pretty-print a local agent trace.

Usage:
  python scripts/open_trace.py --latest
  python scripts/open_trace.py --run-id <run_id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def traces_dir() -> Path:
    return repo_root() / "traces"


def latest_trace_file() -> Path | None:
    d = traces_dir()
    if not d.exists():
        return None
    files = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", action="store_true", help="Open latest trace file")
    ap.add_argument("--run-id", type=str, help="Open traces/<run_id>.jsonl")
    args = ap.parse_args()

    if args.latest:
        p = latest_trace_file()
        if not p:
            raise SystemExit("No traces found.")
    elif args.run_id:
        p = traces_dir() / f"{args.run_id}.jsonl"
    else:
        raise SystemExit("Provide --latest or --run-id")

    if not p.exists():
        raise SystemExit(f"Trace file not found: {p}")

    print(f"TRACE: {p}")
    print("=" * 100)

    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            print(line)
            continue

        ts = ev.get("ts")
        et = ev.get("event")
        if et in ("tool_start", "tool_end"):
            print(f"[{et}] tool={ev.get('tool_name')} id={ev.get('tool_call_id')} ts={ts}")
            if et == "tool_start":
                print("  args:", ev.get("args"))
            else:
                print("  result_preview:", ev.get("result_preview"))
        elif et in ("model_start", "model_end"):
            print(f"[{et}] ts={ts} model={ev.get('model')}")
            if et == "model_start":
                print("  messages_preview:", ev.get("messages_preview"))
            else:
                print("  result_preview:", ev.get("result_preview"))
        else:
            print(ev)
        print("-" * 100)


if __name__ == "__main__":
    main()

