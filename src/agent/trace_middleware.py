"""Agent middleware that writes a local JSONL trace for each run."""

from __future__ import annotations

import uuid
from contextvars import ContextVar
import hashlib
from typing import Any, Callable, TypeVar

from langchain_core.messages import BaseMessage
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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


def _tool_calls_preview(msg: BaseMessage) -> list[dict[str, Any]] | None:
    """Extract tool call info from an AIMessage if present."""
    if not isinstance(msg, AIMessage):
        return None
    tc = getattr(msg, "tool_calls", None)
    if not tc:
        return None
    out: list[dict[str, Any]] = []
    for c in tc:
        # tool call dict: {"name":..., "args":..., "id":...}
        if isinstance(c, dict):
            out.append(
                {
                    "name": c.get("name"),
                    "args": c.get("args"),
                    "id": c.get("id"),
                }
            )
        else:
            out.append({"tool_call": str(c)})
    return out


def _message_event(msg: BaseMessage, *, direction: str) -> dict[str, Any]:
    """Represent a single message as a trace event."""
    content = getattr(msg, "content", None)
    content_str = content if isinstance(content, str) else str(content)
    ev: dict[str, Any] = {
        "event": "message",
        "direction": direction,  # in/out
        "message_type": msg.__class__.__name__,
        "content": content_str,
    }
    # Keep tool call metadata for AI messages
    tc = _tool_calls_preview(msg)
    if tc is not None:
        ev["tool_calls"] = tc
    # ToolMessage has tool_call_id
    if isinstance(msg, ToolMessage):
        ev["tool_call_id"] = getattr(msg, "tool_call_id", None)
    return ev


class LocalTraceMiddleware(AgentMiddleware[Any, Any]):
    """Log model + tool activity to local JSONL files.

    Files: a-share-agent/traces/<run_id>.jsonl
    """

    tools: list[BaseTool] = []

    def __init__(self, *, max_payload_chars: int = 4000) -> None:
        self._writer = get_trace_writer()
        self._max_payload_chars = int(max_payload_chars)
        # Per-run trace id (ContextVar so it works across model/tool calls in the same run)
        self._trace_id_var: ContextVar[str | None] = ContextVar("agent_trace_id", default=None)

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Initialize a per-run trace id as early as possible."""
        try:
            ctx = getattr(runtime, "context", None)
            if isinstance(ctx, dict) and not ctx.get("_trace_id"):
                ctx["_trace_id"] = str(uuid.uuid4())
        except Exception:
            pass
        return None

    def _trace_id_from_state(self, state: Any) -> str | None:
        """Try to derive a stable trace id from the agent state/messages."""
        try:
            if not isinstance(state, dict):
                return None
            msgs = state.get("messages")
            if not isinstance(msgs, list) or not msgs:
                return None
            # Prefer the first HumanMessage id (agent-chat-ui assigns UUIDs)
            for m in msgs:
                if isinstance(m, HumanMessage):
                    mid = getattr(m, "id", None)
                    if mid:
                        return str(mid)
                    content = getattr(m, "content", "")
                    if isinstance(content, str) and content.strip():
                        h = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
                        return f"hmsg_{h}"
            return None
        except Exception:
            return None

    def _trace_id(self, runtime: Any, state: Any | None = None) -> str:
        """Best-effort stable trace id per agent run.

        Priority:
        1) runtime.config.run_id (if present)
        1.5) runtime.config.configurable.thread_id / checkpoint_id (if present)
        2) state-derived id (first human message id/content hash)
        3) ContextVar trace id (best-effort within task)
        4) runtime.context._trace_id (we set this on first access)
        3) generate a UUID and store it in runtime.context
        """
        # 2) state-derived id (best for LangGraph where execution may hop tasks)
        if state is not None:
            sid = self._trace_id_from_state(state)
            if sid:
                self._trace_id_var.set(sid)
                # stash into runtime.context if possible
                try:
                    ctx = getattr(runtime, "context", None)
                    if isinstance(ctx, dict):
                        ctx.setdefault("_trace_id", sid)
                except Exception:
                    pass
                return sid

        # 3) ContextVar (best-effort within same task)
        existing_cv = self._trace_id_var.get()
        if existing_cv:
            return existing_cv

        # 1) run_id if present
        config = getattr(runtime, "config", None)
        if isinstance(config, dict) and config.get("run_id"):
            rid = str(config["run_id"])
            self._trace_id_var.set(rid)
            return rid

        # 1.5) use thread_id/checkpoint_id if available (LangGraph server runs)
        if isinstance(config, dict):
            conf = config.get("configurable")
            if isinstance(conf, dict):
                tid = conf.get("thread_id") or conf.get("checkpoint_id")
                if tid:
                    tid_str = str(tid)
                    self._trace_id_var.set(tid_str)
                    return tid_str

        # 2) context stash
        ctx = getattr(runtime, "context", None)
        if isinstance(ctx, dict):
            existing = ctx.get("_trace_id")
            if existing:
                eid = str(existing)
                self._trace_id_var.set(eid)
                return eid
            new_id = str(uuid.uuid4())
            ctx["_trace_id"] = new_id
            self._trace_id_var.set(new_id)
            return new_id

        # 3) fallback
        new_id = str(uuid.uuid4())
        self._trace_id_var.set(new_id)
        return new_id

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ):
        run_id = self._trace_id(request.runtime, state=getattr(request, "state", None))
        try:
            # Log the latest inbound message (usually HumanMessage or ToolMessage)
            if request.messages:
                self._writer.write_event(
                    run_id,
                    _message_event(request.messages[-1], direction="in"),
                )

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

            # Log each outbound message from the model (usually AIMessage)
            for m in resp.result:
                # Keep only AI/tool messages (but safe to log all)
                self._writer.write_event(run_id, _message_event(m, direction="out"))
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
        run_id = self._trace_id(runtime, state=getattr(request, "state", None))
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

            # Also record the tool message as a message event (so message stream is complete)
            if isinstance(result, ToolMessage):
                # Use truncated content to avoid huge files
                msg_ev = _message_event(result, direction="out")
                if isinstance(msg_ev.get("content"), str) and len(msg_ev["content"]) > self._max_payload_chars:
                    msg_ev["content"] = msg_ev["content"][: self._max_payload_chars] + "..."
                self._writer.write_event(run_id, msg_ev)
        except Exception:
            pass

        return result

