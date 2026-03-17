"""Per-user long-term memory module using mem0.

Provides:
- Automatic memory management: extracts and stores facts from conversations
- Semantic search: retrieves relevant memories for each user query
- Memory tools: LangChain tools for explicit memory operations by the agent

Memory is scoped per user_id, which is injected into LangGraph config.configurable
by the Next.js proxy from the authenticated session.

Architecture:
  Frontend (auth session) → Proxy (injects user_id) → LangGraph (config.configurable.user_id)
  → Memory middleware reads user_id → mem0 operations scoped to that user
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

_logger = logging.getLogger(__name__)

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

# NOTE: LangChain's @tool decorator uses an identity check (`type_ is RunnableConfig`)
# to auto-inject the runtime config.  `Optional[RunnableConfig]` resolves to
# `Union[RunnableConfig, None]` which FAILS that check, so config would never be
# injected.  All tool signatures below therefore use bare `RunnableConfig`.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use environment variables for flexible configuration
MEM0_ENABLED = os.environ.get("MEM0_ENABLED", "true").lower() in ("true", "1", "yes")

# Data directory for persistent memory storage (qdrant on-disk)
_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "mem0_qdrant"


def _get_mem0_config() -> dict:
    """Build mem0 configuration.

    Uses DeepSeek as the LLM (via OpenAI-compatible endpoint) and
    local HuggingFace embeddings to avoid extra API key requirements.
    Qdrant stores vectors to disk for persistence across restarts.
    """
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    # LLM configuration: prefer DeepSeek (already in use), fallback to OpenAI
    if deepseek_key:
        llm_config = {
            "provider": "openai",
            "config": {
                "model": "deepseek-chat",
                "openai_base_url": "https://api.deepseek.com",
                "api_key": deepseek_key,
            },
        }
    elif openai_key:
        llm_config = {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "api_key": openai_key,
            },
        }
    else:
        # Fallback — mem0 default (will fail if no OPENAI_API_KEY)
        llm_config = {}

    # Embedder: prefer OpenAI if available, else use HuggingFace local
    if openai_key:
        embedder_config = {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
                "api_key": openai_key,
            },
        }
    else:
        # bge-small-zh-v1.5 is optimised for Chinese text —
        # the previous all-MiniLM-L6-v2 was English-focused and
        # caused near-zero recall on Chinese memory queries.
        embedder_config = {
            "provider": "huggingface",
            "config": {
                "model": "BAAI/bge-small-zh-v1.5",
            },
        }

    config: dict = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "a_share_agent_memories",
                "path": str(_DATA_DIR),
            },
        },
    }
    if llm_config:
        config["llm"] = llm_config
    if embedder_config:
        config["embedder"] = embedder_config

    return config


# ---------------------------------------------------------------------------
# Singleton memory instance
# ---------------------------------------------------------------------------

_memory = None


def get_memory():
    """Lazily initialise and return the global mem0 Memory instance."""
    global _memory
    if _memory is not None:
        return _memory

    if not MEM0_ENABLED:
        return None

    try:
        from mem0 import Memory

        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        cfg = _get_mem0_config()
        _memory = Memory.from_config(cfg)
        print(f"[memory] mem0 initialised (data: {_DATA_DIR})")
        return _memory
    except ImportError:
        print("[memory] mem0ai not installed — memory features disabled.  pip install mem0ai")
        return None
    except Exception as exc:
        print(f"[memory] Failed to initialise mem0: {exc}")
        return None


# ---------------------------------------------------------------------------
# Helper: extract user_id from LangGraph RunnableConfig
# ---------------------------------------------------------------------------


def get_user_id(config: Optional[RunnableConfig] = None) -> Optional[str]:
    """Extract user_id from LangGraph config.configurable.user_id.

    Falls back to 'dev_user' when running locally without auth,
    so memory features still work during development.
    """
    _fallback = os.environ.get("DEFAULT_USER_ID", "dev_user")
    if config is None:
        _logger.warning("get_user_id called without config — falling back to '%s'", _fallback)
        return _fallback
    configurable = config.get("configurable", {})
    uid = configurable.get("user_id")
    if uid:
        return str(uid)
    _logger.warning("No user_id in config.configurable — falling back to '%s'", _fallback)
    return _fallback


# ---------------------------------------------------------------------------
# Memory operations (called by middleware or tools)
# ---------------------------------------------------------------------------


def search_memories(query: str, user_id: str, limit: int = 5) -> list[dict]:
    """Search memories relevant to *query* for a specific user."""
    mem = get_memory()
    if mem is None:
        return []
    try:
        results = mem.search(query=query, user_id=user_id, limit=limit)
        # mem0 returns {"results": [...]} or a list depending on version
        if isinstance(results, dict) and "results" in results:
            return results["results"]
        if isinstance(results, list):
            return results
        return []
    except Exception as exc:
        print(f"[memory] search error: {exc}")
        return []


def add_memory(messages: list[dict], user_id: str, metadata: dict | None = None) -> dict | None:
    """Store a conversation turn (list of role/content dicts) as a memory."""
    mem = get_memory()
    if mem is None:
        return None
    try:
        result = mem.add(messages, user_id=user_id, metadata=metadata or {})
        return result
    except Exception as exc:
        print(f"[memory] add error: {exc}")
        return None


def get_all_memories(user_id: str) -> list[dict]:
    """Retrieve all stored memories for a user."""
    mem = get_memory()
    if mem is None:
        return []
    try:
        results = mem.get_all(user_id=user_id)
        if isinstance(results, dict) and "results" in results:
            return results["results"]
        if isinstance(results, list):
            return results
        return []
    except Exception as exc:
        print(f"[memory] get_all error: {exc}")
        return []


# ---------------------------------------------------------------------------
# LangChain tool wrappers (agent can call these explicitly)
# ---------------------------------------------------------------------------


@tool
def tool_memory_search(query: str, *, config: RunnableConfig) -> dict:
    """Search your long-term memory for information about the user.

    Use this tool to recall:
    - User's investment portfolio (stocks they hold, positions)
    - User's risk preferences and investment style
    - Previous analysis results or strategies discussed
    - Any facts the user has shared about themselves

    This searches memories stored from previous conversations with this user.

    Args:
        query: Natural language query describing what to look for.
               e.g. "user's portfolio", "risk tolerance", "preferred sectors"
    """
    user_id = get_user_id(config)
    if not user_id:
        return {"memories": [], "note": "No user context — memory not available"}

    memories = search_memories(query, user_id, limit=5)
    return {
        "memories": [m.get("memory", str(m)) for m in memories],
        "count": len(memories),
    }


@tool
def tool_memory_save(content: str, *, config: RunnableConfig) -> dict:
    """Save important information about the user to long-term memory.

    Use this to explicitly remember:
    - User's portfolio changes (bought/sold stocks)
    - User's stated preferences or goals
    - Key analysis findings the user wants to remember
    - Investment strategies the user has decided to follow

    The information will be available in future conversations with this user.

    Args:
        content: The information to remember, as a clear factual statement.
                 e.g. "User holds 1000 shares of 贵州茅台 (600519.SH) bought at ¥1850"
    """
    user_id = get_user_id(config)
    if not user_id:
        return {"saved": False, "note": "No user context — memory not available"}

    messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": f"Noted. I'll remember: {content}"},
    ]
    result = add_memory(messages, user_id)
    return {
        "saved": True,
        "note": "Memory saved successfully. It will be available in future conversations.",
    }


@tool
def tool_memory_list(*, config: RunnableConfig) -> dict:
    """List all stored memories for the current user.

    Returns all facts and information remembered from previous conversations.
    Useful to review what the system knows about the user.
    """
    user_id = get_user_id(config)
    if not user_id:
        return {"memories": [], "note": "No user context — memory not available"}

    memories = get_all_memories(user_id)
    return {
        "memories": [m.get("memory", str(m)) for m in memories],
        "count": len(memories),
    }


# Exported list of memory tools
MEMORY_TOOLS = [
    tool_memory_search,
    tool_memory_save,
    tool_memory_list,
]
