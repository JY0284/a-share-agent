"""Memory middleware – dual-layer context injection.

Layer 1 – **Structured UserProfile** (ALWAYS injected):
  Loads the Pydantic UserProfile from JSON (portfolio, preferences, watchlist,
  strategies).  Fast, deterministic, no LLM / embedding overhead.

Layer 2 – **Soft mem0 memories** (BEST-EFFORT):
  Searches Qdrant for conversational facts (opinions, past discussions).
  Uses Chinese-optimised embeddings (bge-small-zh-v1.5).

Both layers are merged into a context block **prepended to the last
HumanMessage** so the model sees the user's full context without breaking
prefix caching.  The system_message stays 100% static.

The user_id is read from runtime.config.configurable.user_id (injected by the
Next.js proxy from the authenticated session).  Falls back to 'dev_user'
during local development.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage

_logger = logging.getLogger(__name__)
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest

from agent.memory import (
    MEM0_ENABLED,
    search_memories,
    add_memory,
    get_memory,
)
from agent.user_profile import get_or_create_profile, format_full_profile_context
from agent.prompts import get_current_date_block

_DEFAULT_USER_ID = os.environ.get("DEFAULT_USER_ID", "dev_user")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_user_id_from_runtime(runtime: Any) -> str:
    """Extract user_id from runtime.config.configurable.user_id.

    Always returns a usable string (falls back to dev_user).
    """
    try:
        config = getattr(runtime, "config", None)
        if isinstance(config, dict):
            configurable = config.get("configurable", {})
            if isinstance(configurable, dict):
                uid = configurable.get("user_id")
                if uid:
                    return str(uid)
    except Exception:
        pass
    # Fallback: try runtime.context
    try:
        ctx = getattr(runtime, "context", None)
        if isinstance(ctx, dict):
            uid = ctx.get("user_id")
            if uid:
                return str(uid)
        if ctx and hasattr(ctx, "user_id"):
            uid = getattr(ctx, "user_id", None)
            if uid:
                return str(uid)
    except Exception:
        pass
    _logger.warning("No user_id found in runtime — falling back to '%s'", _DEFAULT_USER_ID)
    return _DEFAULT_USER_ID


def _get_last_human_content(messages: list) -> Optional[str]:
    """Find the latest human message content."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                return " ".join(parts) if parts else None
        if isinstance(msg, dict) and msg.get("type") == "human":
            c = msg.get("content")
            if isinstance(c, str):
                return c
    return None


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class MemoryMiddleware(AgentMiddleware[Any, Any]):
    """Dual-layer memory middleware: structured UserProfile + soft mem0.

    Inherits from AgentMiddleware and intercepts model calls to inject
    the user's full context into the system message.
    """

    tools: list = []  # Required by AgentMiddleware interface

    def __init__(self, max_memories: int = 5):
        self.max_memories = max_memories

    # ------------------------------------------------------------------
    # Build the combined context block
    # ------------------------------------------------------------------

    _CONTEXT_MARKER = "[User Context]"

    def _build_context_block(
        self,
        profile_text: str | None,
        memories: list[dict] | None,
    ) -> str | None:
        """Merge live date + structured profile + soft memories into one text block."""
        parts: list[str] = []

        # Layer 0 — live date (changes once per day — excellent for caching)
        parts.append(get_current_date_block())

        # Layer 1 — structured profile (always present if profile exists)
        if profile_text:
            parts.append(profile_text)

        # Layer 2 — soft memories from mem0
        if memories:
            mem_lines = []
            for mem in memories:
                text = mem.get("memory", str(mem)) if isinstance(mem, dict) else str(mem)
                mem_lines.append(f"  - {text}")
            if mem_lines:
                parts.append(
                    "\n## 📝 Conversational Memory (soft – from previous chats)\n"
                    + "\n".join(mem_lines)
                )

        if not parts:
            return None

        return "\n".join(parts)

    def _load_profile_text(self, user_id: str) -> str | None:
        """Load structured UserProfile and format for injection."""
        try:
            profile = get_or_create_profile(user_id)
            text = format_full_profile_context(profile)
            return text if text else None
        except Exception as exc:
            print(f"[memory-middleware] profile load error: {exc}")
            return None

    def _inject_context(
        self, request: ModelRequest, context_block: str
    ) -> ModelRequest:
        """Prepend context to the **last HumanMessage**, keeping everything else untouched.

        Why: Model providers (DeepSeek, OpenAI, etc.) cache based on the
        token prefix.  The prefix is: static system_prompt → tools → earlier
        messages.  By injecting dynamic data (date, profile, memories) into
        the last HumanMessage — the only thing that changes — the entire
        preceding conversation stays a stable, cacheable prefix.
        """
        messages = list(request.messages or [])

        # Find the last HumanMessage
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, HumanMessage):
                content = msg.content
                # Guard against double-injection
                if isinstance(content, str) and self._CONTEXT_MARKER in content:
                    return request
                elif isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                            and isinstance(block.get("text"), str)
                            and self._CONTEXT_MARKER in block.get("text", "")
                        ):
                            return request

                prefix = f"{self._CONTEXT_MARKER}\n{context_block.strip()}\n---\n"
                if isinstance(content, str):
                    new_content = prefix + content
                elif isinstance(content, list):
                    # Multimodal: prepend as a text block
                    new_content = [{"type": "text", "text": prefix}, *content]
                else:
                    new_content = prefix + str(content)

                new_msg = HumanMessage(content=new_content)  # type: ignore[arg-type]
                new_messages = messages[:i] + [new_msg] + messages[i + 1:]
                return request.override(messages=new_messages)

        # No HumanMessage found — should not happen, but fall back safely
        return request

    # ------------------------------------------------------------------
    # Async wrapper
    # ------------------------------------------------------------------

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[Any]],
    ) -> Any:
        """Inject structured profile + soft memories before each model call."""
        user_id = _get_user_id_from_runtime(request.runtime)

        # Layer 1 — structured profile (sync, fast JSON read)
        profile_text = await asyncio.to_thread(self._load_profile_text, user_id)

        # Layer 2 — soft mem0 (best-effort)
        memories: list[dict] = []
        if MEM0_ENABLED and get_memory() is not None:
            latest_human = _get_last_human_content(list(request.messages or []))
            if latest_human:
                try:
                    memories = await asyncio.to_thread(
                        search_memories, latest_human, user_id, limit=self.max_memories
                    )
                except Exception as exc:
                    print(f"[memory-middleware] mem0 search error (non-fatal): {exc}")

        context_block = self._build_context_block(profile_text, memories)
        if not context_block:
            return await handler(request)

        return await handler(self._inject_context(request, context_block))

    # ------------------------------------------------------------------
    # Sync wrapper
    # ------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> Any:
        """Sync version of dual-layer context injection."""
        user_id = _get_user_id_from_runtime(request.runtime)

        # Layer 1 — structured profile
        profile_text = self._load_profile_text(user_id)

        # Layer 2 — soft mem0
        memories: list[dict] = []
        if MEM0_ENABLED and get_memory() is not None:
            latest_human = _get_last_human_content(list(request.messages or []))
            if latest_human:
                try:
                    memories = search_memories(
                        latest_human, user_id, limit=self.max_memories
                    )
                except Exception as exc:
                    print(f"[memory-middleware] mem0 search error (non-fatal): {exc}")

        context_block = self._build_context_block(profile_text, memories)
        if not context_block:
            return handler(request)

        return handler(self._inject_context(request, context_block))


# ---------------------------------------------------------------------------
# Post-processing callback (unchanged, but with dev_user fallback)
# ---------------------------------------------------------------------------


class MemoryStorageCallback:
    """Post-processing callback to store conversation in memory.

    Call this after the agent produces a response to persist the
    conversation turn into mem0.
    """

    @staticmethod
    def store_turn(
        user_message: str,
        assistant_message: str,
        user_id: str | None = None,
    ) -> None:
        """Store a single conversation turn as a memory."""
        if not MEM0_ENABLED or get_memory() is None:
            return
        user_id = user_id or _DEFAULT_USER_ID
        if not user_message:
            return

        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message[:2000]},  # truncate
        ]

        try:
            add_memory(messages, user_id)
        except Exception as exc:
            print(f"[memory-middleware] Failed to store turn: {exc}")
