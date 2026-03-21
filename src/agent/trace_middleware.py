"""Agent middleware that writes a local JSONL trace for each run.

Trace format: clean **messages** (one JSON object per line), similar to the
OpenAI chat-completion messages format:

    {"role": "system", "content": "..."}
    {"role": "user",   "content": [...]}
    {"role": "assistant", "content": "...", "tool_calls": [...], "model": "...", "usage": {...}, "cost": {...}}
    {"role": "tool",   "name": "...",  "content": "...", "tool_call_id": "..."}
    {"role": "assistant", "content": "...", "model": "...", "usage": {...}, "cost": {...}}

Each line is self-contained JSON with a ``ts`` timestamp.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
import hashlib
from typing import Any, Callable, Sequence, TypeVar

from langchain_core.messages import BaseMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)

from agent.sandbox import set_python_session_id, set_thread_id
from agent.trace import get_trace_writer
from agent.usage_cost import compute_usage_and_cost, load_pricing

TContext = TypeVar("TContext")
TState = TypeVar("TState")

# ---------------------------------------------------------------------------
# Per-run state: track which messages have already been logged so that
# repeated model calls (tool-loop iterations) don't duplicate them.
# ---------------------------------------------------------------------------
_logged_ids: ContextVar[set | None] = ContextVar("_trace_logged_ids", default=None)


def _get_logged_ids() -> set:
    s = _logged_ids.get(None)
    if s is None:
        s = set()
        _logged_ids.set(s)
    return s


# ---------------------------------------------------------------------------
# Convert LangChain messages → clean trace messages
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int) -> str:
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "...(truncated)"
    return text


def _msg_to_trace(msg: BaseMessage, *, max_chars: int = 0) -> dict[str, Any]:
    """Convert a LangChain message to a clean trace dict with role/content."""
    content = getattr(msg, "content", None)
    content_out: Any
    if isinstance(content, str):
        content_out = _truncate(content, max_chars)
    elif isinstance(content, list):
        # Multimodal content blocks — keep structure but truncate text blocks
        out_blocks: list[Any] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                out_blocks.append({**block, "text": _truncate(block.get("text", ""), max_chars)})
            elif isinstance(block, dict) and block.get("type") in ("image", "image_url"):
                # Don't dump base64 into the trace; just note it
                out_blocks.append({"type": "image", "mimeType": block.get("mimeType", "image/*")})
            elif isinstance(block, str):
                out_blocks.append(_truncate(block, max_chars))
            else:
                out_blocks.append(block)
        content_out = out_blocks
    else:
        content_out = str(content) if content is not None else ""

    # Determine role
    if isinstance(msg, SystemMessage):
        role = "system"
    elif isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, AIMessage):
        role = "assistant"
    elif isinstance(msg, ToolMessage):
        role = "tool"
    else:
        role = msg.__class__.__name__.lower().replace("message", "")

    ev: dict[str, Any] = {"role": role, "content": content_out}

    # Tool calls on assistant messages
    if isinstance(msg, AIMessage):
        tc = getattr(msg, "tool_calls", None)
        if tc:
            ev["tool_calls"] = [
                {"name": c.get("name"), "args": c.get("args"), "id": c.get("id")}
                if isinstance(c, dict) else {"tool_call": str(c)}
                for c in tc
            ]

    # tool_call_id on ToolMessage
    if isinstance(msg, ToolMessage):
        ev["tool_call_id"] = getattr(msg, "tool_call_id", None)
        ev["name"] = getattr(msg, "name", None)

    return ev


def _extract_result_messages(resp: Any) -> list[BaseMessage]:
    """Extract messages from ModelCallResult (ModelResponse | AIMessage)."""
    if hasattr(resp, "result") and isinstance(resp.result, list):
        return resp.result
    if isinstance(resp, BaseMessage):
        return [resp]
    return []


def _compute_usage_cost_for_msg(msg: AIMessage, model_name: str | None) -> dict[str, Any]:
    """Compute usage/cost for a single AIMessage and attach to its kwargs.

    Returns {"usage": {...}, "cost": {...}} or empty dict.
    """
    info = compute_usage_and_cost(msg, model_name=model_name)
    usage = info.get("usage")
    cost = info.get("cost")
    result: dict[str, Any] = {}

    if usage:
        result["usage"] = usage
        # Attach to message objects for billing proxy
        for attr in ("additional_kwargs", "response_metadata"):
            container = getattr(msg, attr, None)
            if not isinstance(container, dict):
                container = {}
                try:
                    setattr(msg, attr, container)
                except Exception:
                    continue
            container["usage"] = usage
    if cost:
        result["cost"] = cost
        for attr in ("additional_kwargs", "response_metadata"):
            container = getattr(msg, attr, None)
            if isinstance(container, dict):
                container["cost"] = cost

    return result


def _collect_vision_costs() -> tuple[list[dict], float]:
    """Consume pending vision costs (from VisionMiddleware) and compute total."""
    try:
        from agent.vision_middleware import consume_pending_vision_costs
    except ImportError:
        return [], 0.0
    vision_costs = consume_pending_vision_costs()
    if not vision_costs:
        return [], 0.0
    vision_total = sum(
        (vc.get("cost") or {}).get("total", 0) for vc in vision_costs
    )
    return vision_costs, vision_total


def _collect_image_info() -> dict | None:
    """Consume original image metadata (set by VisionMiddleware before stripping)."""
    try:
        from agent.vision_middleware import consume_pending_image_info
    except ImportError:
        return None
    return consume_pending_image_info()


def _apply_vision_costs_to_msg(
    vision_costs: list[dict],
    vision_total: float,
    result_messages: list[BaseMessage],
) -> None:
    """Attach vision costs to the last AIMessage for billing."""
    if not vision_costs or vision_total <= 0:
        return
    for m in reversed(result_messages):
        if not isinstance(m, AIMessage):
            continue
        for attr_name in ("additional_kwargs", "response_metadata"):
            container = getattr(m, attr_name, None)
            if not isinstance(container, dict):
                continue
            if "cost" in container and isinstance(container["cost"], dict):
                container["cost"]["total"] = round(
                    float(container["cost"].get("total", 0)) + vision_total, 6
                )
            else:
                container["cost"] = {"currency": "RMB", "total": round(vision_total, 6)}
        break


# ---------------------------------------------------------------------------
# Middleware class
# ---------------------------------------------------------------------------

class LocalTraceMiddleware(AgentMiddleware[Any, Any]):
    """Log model + tool activity as clean messages to local JSONL files.

    Trace directory: a-share-agent-traces/<user_name>/<datetime>_<run_id>.jsonl
    """

    tools: list[BaseTool] = []

    def __init__(self, *, max_payload_chars: int = 4000) -> None:
        self._writer = get_trace_writer()
        self._max_payload_chars = int(max_payload_chars)
        self._trace_id_var: ContextVar[str | None] = ContextVar("agent_trace_id", default=None)
        # Track whether system prompt has been logged for this run
        self._system_logged: ContextVar[set | None] = ContextVar("_trace_system_logged", default=None)
        try:
            load_pricing()
        except Exception:
            pass

    def _get_system_logged(self) -> set:
        s = self._system_logged.get(None)
        if s is None:
            s = set()
            self._system_logged.set(s)
        return s

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        try:
            ctx = getattr(runtime, "context", None)
            if isinstance(ctx, dict) and not ctx.get("_trace_id"):
                ctx["_trace_id"] = str(uuid.uuid4())
        except Exception:
            pass
        return None

    def _trace_id_from_state(self, state: Any) -> str | None:
        try:
            if not isinstance(state, dict):
                return None
            msgs = state.get("messages")
            if not isinstance(msgs, list) or not msgs:
                return None
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
        if state is not None:
            sid = self._trace_id_from_state(state)
            if sid:
                self._trace_id_var.set(sid)
                try:
                    ctx = getattr(runtime, "context", None)
                    if isinstance(ctx, dict):
                        ctx.setdefault("_trace_id", sid)
                except Exception:
                    pass
                return sid

        existing_cv = self._trace_id_var.get()
        if existing_cv:
            return existing_cv

        config = getattr(runtime, "config", None)
        if isinstance(config, dict) and config.get("run_id"):
            rid = str(config["run_id"])
            self._trace_id_var.set(rid)
            return rid

        if isinstance(config, dict):
            conf = config.get("configurable")
            if isinstance(conf, dict):
                tid = conf.get("thread_id") or conf.get("checkpoint_id")
                if tid:
                    tid_str = str(tid)
                    self._trace_id_var.set(tid_str)
                    return tid_str

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

        new_id = str(uuid.uuid4())
        self._trace_id_var.set(new_id)
        return new_id

    def _get_user_display_name(self, runtime: Any) -> str | None:
        """Extract a human-readable user identifier for trace directories.

        Priority: user_name > user_email > user_id (UUID fallback).
        """
        try:
            config = getattr(runtime, "config", None)
            if isinstance(config, dict):
                configurable = config.get("configurable", {})
                if isinstance(configurable, dict):
                    # Prefer human-readable name
                    name = configurable.get("user_name")
                    if name:
                        return str(name)
                    email = configurable.get("user_email")
                    if email:
                        return str(email)
                    uid = configurable.get("user_id")
                    if uid:
                        return str(uid)
        except Exception:
            pass
        # Fallback: runtime.context
        try:
            ctx = getattr(runtime, "context", None)
            if isinstance(ctx, dict):
                uid = ctx.get("user_id")
                if uid:
                    return str(uid)
        except Exception:
            pass
        return None

    def _get_thread_id(self, runtime: Any) -> str | None:
        config = getattr(runtime, "config", None)
        if config is None:
            return None
        if isinstance(config, dict):
            conf = config.get("configurable")
        else:
            conf = getattr(config, "configurable", None)
        if conf is None:
            return None
        if isinstance(conf, dict):
            tid = conf.get("thread_id")
        else:
            tid = getattr(conf, "thread_id", None)
        return str(tid) if tid else None

    # ------------------------------------------------------------------
    # Setup helpers (shared by sync/async)
    # ------------------------------------------------------------------

    def _setup_run(self, runtime: Any, state: Any | None) -> str:
        """Common run setup: derive IDs, register user, return run_id."""
        run_id = self._trace_id(runtime, state=state)
        set_python_session_id(run_id)
        actual_thread_id = self._get_thread_id(runtime)
        if actual_thread_id:
            set_thread_id(actual_thread_id)
        user_display = self._get_user_display_name(runtime)
        if user_display:
            self._writer.set_user_for_run(run_id, user_display)
        return run_id

    # ------------------------------------------------------------------
    # Log new input messages (avoiding duplicates across agent-loop iters)
    # ------------------------------------------------------------------

    def _log_new_input_messages(self, run_id: str, messages: Sequence[BaseMessage] | None, *, sync: bool = True) -> list:
        """Log any input messages not yet logged. Returns async coroutines if sync=False."""
        if not messages:
            return []
        logged = _get_logged_ids()
        system_logged = self._get_system_logged()
        # Collect image info once (consumed per model call)
        image_info = _collect_image_info()
        coros = []
        for msg in messages:
            msg_id = id(msg)
            if msg_id in logged:
                continue
            logged.add(msg_id)

            # Log system messages (the system prompt)
            if isinstance(msg, SystemMessage):
                if run_id not in system_logged:
                    system_logged.add(run_id)
                    ev = _msg_to_trace(msg, max_chars=0)  # full system prompt
                    if sync:
                        self._writer.write_event(run_id, ev)
                    else:
                        coros.append(self._writer.awrite_event(run_id, ev))
                continue

            ev = _msg_to_trace(msg, max_chars=self._max_payload_chars)
            # Attach original image info to user messages (images already stripped by VisionMiddleware)
            if isinstance(msg, HumanMessage) and image_info:
                ev["original_images"] = image_info
                image_info = None  # Only attach once
            if sync:
                self._writer.write_event(run_id, ev)
            else:
                coros.append(self._writer.awrite_event(run_id, ev))
        return coros

    # ------------------------------------------------------------------
    # Model call wrappers
    # ------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ):
        run_id = self._setup_run(request.runtime, getattr(request, "state", None))
        model_name = getattr(request.model, "model", None) or getattr(request.model, "model_name", None)

        try:
            self._log_new_input_messages(run_id, request.messages, sync=True)
        except Exception:
            pass

        resp = handler(request)
        result_messages = _extract_result_messages(resp)

        # Log assistant messages with embedded usage/cost
        try:
            vision_costs, vision_total = _collect_vision_costs()
            _apply_vision_costs_to_msg(vision_costs, vision_total, result_messages)

            for m in result_messages:
                ev = _msg_to_trace(m, max_chars=self._max_payload_chars)
                if isinstance(m, AIMessage):
                    ev["model"] = model_name or m.__class__.__name__
                    uc = _compute_usage_cost_for_msg(m, model_name)
                    ev.update(uc)
                    if vision_costs:
                        ev["vision"] = vision_costs
                self._writer.write_event(run_id, ev)
                _get_logged_ids().add(id(m))
        except Exception as exc:
            print(f"[trace] message logging failed: {type(exc).__name__}: {exc}")

        return resp

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> ModelResponse:
        run_id = self._setup_run(request.runtime, getattr(request, "state", None))
        model_name = getattr(request.model, "model", None) or getattr(request.model, "model_name", None)

        try:
            coros = self._log_new_input_messages(run_id, request.messages, sync=False)
            for c in coros:
                await c
        except Exception:
            pass

        resp = await handler(request)
        result_messages = _extract_result_messages(resp)

        try:
            vision_costs, vision_total = _collect_vision_costs()
            _apply_vision_costs_to_msg(vision_costs, vision_total, result_messages)

            for m in result_messages:
                ev = _msg_to_trace(m, max_chars=self._max_payload_chars)
                if isinstance(m, AIMessage):
                    ev["model"] = model_name or m.__class__.__name__
                    uc = _compute_usage_cost_for_msg(m, model_name)
                    ev.update(uc)
                    if vision_costs:
                        ev["vision"] = vision_costs
                await self._writer.awrite_event(run_id, ev)
                _get_logged_ids().add(id(m))
        except Exception as exc:
            print(f"[trace] async message logging failed: {type(exc).__name__}: {exc}")

        return resp

    # ------------------------------------------------------------------
    # Tool call wrappers
    # ------------------------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ):
        runtime = getattr(request, "runtime", None)
        run_id = self._setup_run(runtime, getattr(request, "state", None))

        result = handler(request)

        # Log tool result as a clean message
        try:
            tool_call = request.tool_call or {}
            name = tool_call.get("name")
            tool_call_id = tool_call.get("id") or getattr(runtime, "tool_call_id", None)

            content = getattr(result, "content", result)
            content_str = content if isinstance(content, str) else str(content)
            content_str = _truncate(content_str, self._max_payload_chars)

            ev: dict[str, Any] = {
                "role": "tool",
                "name": name,
                "tool_call_id": tool_call_id,
                "content": content_str,
            }
            self._writer.write_event(run_id, ev)
            if isinstance(result, ToolMessage):
                _get_logged_ids().add(id(result))
        except Exception as exc:
            print(f"[trace] tool logging failed: {type(exc).__name__}: {exc}")

        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        runtime = getattr(request, "runtime", None)
        run_id = self._setup_run(runtime, getattr(request, "state", None))

        result = await handler(request)

        try:
            tool_call = request.tool_call or {}
            name = tool_call.get("name")
            tool_call_id = tool_call.get("id") or getattr(runtime, "tool_call_id", None)

            content = getattr(result, "content", result)
            content_str = content if isinstance(content, str) else str(content)
            content_str = _truncate(content_str, self._max_payload_chars)

            ev: dict[str, Any] = {
                "role": "tool",
                "name": name,
                "tool_call_id": tool_call_id,
                "content": content_str,
            }
            await self._writer.awrite_event(run_id, ev)
            if isinstance(result, ToolMessage):
                _get_logged_ids().add(id(result))
        except Exception as exc:
            print(f"[trace] async tool logging failed: {type(exc).__name__}: {exc}")

        return result

