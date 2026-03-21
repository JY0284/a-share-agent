"""Local trace recorder for LangGraph agent runs.

Goal:
- Save each run's model/tool activity locally for debugging.
- Keep it simple and dependency-free.

Implementation:
- A middleware logs conversation messages as JSONL (one message per line)
- Files are saved under: a-share-agent-traces/<user_name>/<datetime>_<run_id>.jsonl
- Anonymous runs go to: a-share-agent-traces/_anonymous/...

Note:
  ``langgraph dev`` enables BlockBuster which **blocks** synchronous I/O
  inside async functions.  All writes from async middleware must therefore
  go through ``awrite_event`` (which delegates to a thread-pool) rather
  than the synchronous ``write_event``.
"""

from __future__ import annotations

import asyncio
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
    _run_user_ids: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def set_user_for_run(self, run_id: str, user_display: str) -> None:
        """Associate a display name with a run so traces go into per-user dirs.

        ``user_display`` should be a human-readable name (e.g. "yu") or email
        (e.g. "249582269@qq.com").  Falls back to UUID if neither is available.

        Must be called before the first ``path_for_run`` / ``write_event``
        for that run_id.  If the path was already created, this is a no-op.
        """
        if run_id not in self._run_user_ids:
            self._run_user_ids[run_id] = user_display

    def path_for_run(self, run_id: str) -> Path:
        """Return (and cache) the trace file path for a run.

        If a user display name was registered via ``set_user_for_run``,
        the trace file is placed under ``<trace_dir>/<name>/<datetime>_<run_id>.jsonl``.
        Otherwise it falls back to ``<trace_dir>/_anonymous/...``.
        """
        if run_id in self._run_paths:
            return self._run_paths[run_id]
        safe = "".join(ch for ch in str(run_id) if ch.isalnum() or ch in ("-", "_"))
        if not safe:
            safe = f"run_{int(time.time())}"
        dt_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Per-user subdirectory
        user_id = self._run_user_ids.get(run_id)
        if user_id:
            safe_user = "".join(ch for ch in str(user_id) if ch.isalnum() or ch in ("-", "_", "@", "."))
        else:
            safe_user = "_anonymous"
        user_dir = self.trace_dir / safe_user
        user_dir.mkdir(parents=True, exist_ok=True)

        path = user_dir / f"{dt_prefix}_{safe}.jsonl"
        self._run_paths[run_id] = path
        return path

    # -- synchronous (for direct / sync invocation) --

    def write_event(self, run_id: str, event: dict[str, Any]) -> None:
        p = self.path_for_run(run_id)
        payload = {
            "ts": time.time(),
            **event,
        }
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")

    # -- async-safe (for awrap_model_call / awrap_tool_call) --

    def _write_event_sync(self, run_id: str, event: dict[str, Any]) -> None:
        """Internal sync helper used by ``awrite_event``."""
        self.write_event(run_id, event)

    async def awrite_event(self, run_id: str, event: dict[str, Any]) -> None:
        """Async version of ``write_event``.

        Delegates the blocking file I/O to a thread-pool so that
        BlockBuster (used by ``langgraph dev``) does not raise.
        """
        await asyncio.to_thread(self._write_event_sync, run_id, event)


def get_trace_writer() -> TraceWriter:
    env = os.environ.get("AGENT_TRACE_DIR")
    trace_dir = Path(env) if env else _default_trace_dir()
    return TraceWriter(trace_dir=trace_dir)

