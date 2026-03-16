"""Memory middleware for automatic memory retrieval and storage.

This middleware:
1. BEFORE the model is called: searches mem0 for relevant memories based on the
   user's latest message, and injects them as context in the system prompt.
2. AFTER the agent runs: stores the conversation turn in mem0 for future recall.

The user_id is read from runtime.config.configurable.user_id (injected by the
Next.js proxy from the authenticated session).
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest

from agent.memory import (
    MEM0_ENABLED,
    search_memories,
    add_memory,
    get_memory,
)


def _get_user_id_from_runtime(runtime: Any) -> Optional[str]:
    """Extract user_id from runtime.config.configurable.user_id."""
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
    return None


def _get_last_human_content(messages: list) -> Optional[str]:
    """Find the latest human message content."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                return " ".join(parts) if parts else None
        # Also handle dict-style messages
        if isinstance(msg, dict) and msg.get("type") == "human":
            c = msg.get("content")
            if isinstance(c, str):
                return c
    return None


class MemoryMiddleware(AgentMiddleware[Any, Any]):
    """LangGraph middleware that provides automatic per-user memory.

    Inherits from AgentMiddleware and uses awrap_model_call to inject
    relevant memories into the system message before each model call.
    """

    tools: list = []  # Required by AgentMiddleware interface

    def __init__(self, max_memories: int = 5):
        self.max_memories = max_memories

    # ------------------------------------------------------------------
    # Model call wrapper: inject memories before each model call
    # ------------------------------------------------------------------

    def _build_memory_block(self, memories: list[dict]) -> Optional[str]:
        """Build a memory context block from a list of memory dicts."""
        memory_lines: list[str] = []
        for mem in memories:
            text = mem.get("memory", str(mem)) if isinstance(mem, dict) else str(mem)
            memory_lines.append(f"  - {text}")

        if not memory_lines:
            return None

        return (
            "\n\n## 📝 User Memory (from previous conversations)\n"
            "The following facts are remembered about this user. "
            "Use them to personalise your response:\n"
            + "\n".join(memory_lines)
            + "\n"
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[Any]],
    ) -> Any:
        """Inject relevant memories into the system message before the model call."""
        if not MEM0_ENABLED or get_memory() is None:
            return await handler(request)

        user_id = _get_user_id_from_runtime(request.runtime)
        if not user_id:
            return await handler(request)

        messages = list(request.messages or [])
        latest_human = _get_last_human_content(messages)
        if not latest_human:
            return await handler(request)

        # Search for relevant memories (run in thread to avoid blocking)
        memories = await asyncio.to_thread(
            search_memories, latest_human, user_id, limit=self.max_memories
        )
        if not memories:
            return await handler(request)

        memory_block = self._build_memory_block(memories)
        if not memory_block:
            return await handler(request)

        # Inject memory block into the system message
        if request.system_message is not None:
            new_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": memory_block},
            ]
        else:
            new_content = [{"type": "text", "text": memory_block.strip()}]

        new_system_message = SystemMessage(content=new_content)
        return await handler(request.override(system_message=new_system_message))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> Any:
        """Sync version: inject relevant memories into the system message."""
        if not MEM0_ENABLED or get_memory() is None:
            return handler(request)

        user_id = _get_user_id_from_runtime(request.runtime)
        if not user_id:
            return handler(request)

        messages = list(request.messages or [])
        latest_human = _get_last_human_content(messages)
        if not latest_human:
            return handler(request)

        # Search for relevant memories (sync)
        memories = search_memories(latest_human, user_id, limit=self.max_memories)
        if not memories:
            return handler(request)

        memory_block = self._build_memory_block(memories)
        if not memory_block:
            return handler(request)

        # Inject memory block into the system message
        if request.system_message is not None:
            new_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": memory_block},
            ]
        else:
            new_content = [{"type": "text", "text": memory_block.strip()}]

        new_system_message = SystemMessage(content=new_content)
        return handler(request.override(system_message=new_system_message))


class MemoryStorageCallback:
    """Post-processing callback to store conversation in memory.

    Call this after the agent produces a response to persist the
    conversation turn into mem0.
    """

    @staticmethod
    def store_turn(
        user_message: str,
        assistant_message: str,
        user_id: str,
    ) -> None:
        """Store a single conversation turn as a memory."""
        if not MEM0_ENABLED or get_memory() is None:
            return
        if not user_id or not user_message:
            return

        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message[:2000]},  # truncate
        ]

        try:
            add_memory(messages, user_id)
        except Exception as exc:
            print(f"[memory-middleware] Failed to store turn: {exc}")
