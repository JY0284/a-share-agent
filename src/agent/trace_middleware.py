"""Agent middleware that writes a local JSONL trace for each run."""

from __future__ import annotations

import uuid
from typing import Any, Callable, TypeVar

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)

from agent.trace import get_trace_writer

TContext = TypeVar("TContext")
TState = TypeVar("TState")


def _safe_messages_preview(messages: list[BaseMessage], *, max_chars: int = 2000) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    used = 0
    for m in messages:
        content = getattr(m, "content", None)
        content_str = content if isinstance(content, str) else str(content)
        remain = max(0, max_chars - used)
        if remain <= 0:
            break
        snippet = content_str[:remain]
        used += len(snippet)
        out.append({"type": m.__class__.__name__, "content": snippet})
    return out


class LocalTraceMiddleware(AgentMiddleware[Any, Any]):
    """Log model + tool activity to local JSONL files.

    Files: a-share-agent/traces/<run_id>.jsonl
    """

    tools: list[BaseTool] = []

    def __init__(self, *, max_payload_chars: int = 4000) -> None:
        self._writer = get_trace_writer()
        self._max_payload_chars = int(max_payload_chars)

    def _trace_id(self, runtime: Any) -> str:
        """Best-effort stable trace id per agent run.

        Priority:
        1) runtime.config.run_id (if present)
        2) runtime.context._trace_id (we set this on first access)
        3) generate a UUID and store it in runtime.context
        """
        # 1) run_id if present
        config = getattr(runtime, "config", None)
        if isinstance(config, dict) and config.get("run_id"):
            return str(config["run_id"])

        # 2) context stash
        ctx = getattr(runtime, "context", None)
        if isinstance(ctx, dict):
            existing = ctx.get("_trace_id")
            if existing:
                return str(existing)
            new_id = str(uuid.uuid4())
            ctx["_trace_id"] = new_id
            return new_id

        # 3) fallback
        return str(uuid.uuid4())

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ):
        run_id = self._trace_id(request.runtime)
        try:
            self._writer.write_event(
                run_id,
                {
                    "event": "model_start",
                    "model": getattr(request.model, "model", None) or request.model.__class__.__name__,
                    "tool_count": len(request.tools or []),
                    "messages_preview": _safe_messages_preview(request.messages),
                },
            )
        except Exception:
            # tracing should never break the agent
            pass

        resp = handler(request)

        try:
            # response is list[BaseMessage]; preview for size
            self._writer.write_event(
                run_id,
                {
                    "event": "model_end",
                    "result_preview": _safe_messages_preview(resp.result),
                    "structured_response": resp.structured_response,
                },
            )
        except Exception:
            pass

        return resp

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ):
        # ToolRuntime is from langgraph; includes config/tool_call_id
        runtime = getattr(request, "runtime", None)
        run_id = self._trace_id(runtime)
        tool_call = request.tool_call or {}
        name = tool_call.get("name")
        args = tool_call.get("args")
        tool_call_id = tool_call.get("id") or getattr(runtime, "tool_call_id", None)

        try:
            self._writer.write_event(
                run_id,
                {
                    "event": "tool_start",
                    "tool_name": name,
                    "tool_call_id": tool_call_id,
                    "args": args,
                },
            )
        except Exception:
            pass

        result = handler(request)

        # Tool result is typically a ToolMessage; store a compact preview
        try:
            content = getattr(result, "content", result)
            content_str = content if isinstance(content, str) else str(content)
            if len(content_str) > self._max_payload_chars:
                content_str = content_str[: self._max_payload_chars] + "..."

            self._writer.write_event(
                run_id,
                {
                    "event": "tool_end",
                    "tool_name": name,
                    "tool_call_id": tool_call_id,
                    "result_preview": content_str,
                },
            )
        except Exception:
            pass

        return result

