"""Local trace recorder for LangGraph agent runs.

Goal:
- Save each run's model/tool activity locally for debugging.
- Keep it simple and dependency-free.

Implementation:
- A middleware logs model calls + tool calls as JSONL (one event per line)
- Files are saved under: a-share-agent/traces/<run_id>.jsonl
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _default_trace_dir() -> Path:
    # a-share-agent/src/agent/trace.py -> parents[2] is repo root
    return Path(__file__).resolve().parents[2] / "traces"


def _json_default(o: Any) -> str:
    # fallback serialization for unknown objects
    return str(o)


@dataclass
class TraceWriter:
    trace_dir: Path

    def __post_init__(self) -> None:
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def path_for_run(self, run_id: str) -> Path:
        safe = "".join(ch for ch in str(run_id) if ch.isalnum() or ch in ("-", "_"))
        if not safe:
            safe = f"run_{int(time.time())}"
        return self.trace_dir / f"{safe}.jsonl"

    def write_event(self, run_id: str, event: dict[str, Any]) -> None:
        p = self.path_for_run(run_id)
        payload = {
            "ts": time.time(),
            **event,
        }
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def get_trace_writer() -> TraceWriter:
    env = os.environ.get("AGENT_TRACE_DIR")
    trace_dir = Path(env) if env else _default_trace_dir()
    return TraceWriter(trace_dir=trace_dir)

