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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _default_trace_dir() -> Path:
    # Save traces in parent dir of repo to avoid file watcher
    # a-share-agent/src/agent/trace.py -> parents[2] is repo root
    return Path(__file__).resolve().parents[3] / "a-share-agent-traces"


def _json_default(o: Any) -> str:
    # fallback serialization for unknown objects
    return str(o)


@dataclass
class TraceWriter:
    trace_dir: Path
    _run_paths: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def path_for_run(self, run_id: str) -> Path:
        """Return (and cache) the trace file path for a run.

        The first call creates a datetime-prefixed filename so traces sort
        chronologically.  Subsequent calls for the same run_id return the
        same path so the datetime doesn't shift mid-run.
        """
        if run_id in self._run_paths:
            return self._run_paths[run_id]
        safe = "".join(ch for ch in str(run_id) if ch.isalnum() or ch in ("-", "_"))
        if not safe:
            safe = f"run_{int(time.time())}"
        dt_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.trace_dir / f"{dt_prefix}_{safe}.jsonl"
        self._run_paths[run_id] = path
        return path

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

