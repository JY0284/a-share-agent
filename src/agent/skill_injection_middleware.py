"""Middleware for auto-injecting relevant skill content based on user query.

This middleware runs BEFORE the agent processes the query. It:
1. Analyzes the user's question
2. Selects the most relevant skill (if any)
3. Appends the skill content to the user message (not as a separate SystemMessage)

This allows the agent to see relevant skill guidance when writing code,
while keeping the system prompt completely static for LLM provider prompt caching.

NOTE: We append to the user message (not inject a new SystemMessage) to ensure
the system prompt never changes throughout the conversation thread.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from langchain_core.messages import HumanMessage
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest

from agent.skills import select_top_skill_for_query


# Configuration
MAX_SKILL_INJECT_CHARS = 3000  # Maximum characters of skill content to inject
MIN_SKILL_SCORE = 3  # Minimum relevance score to inject a skill (matches select_top_skill_for_query)


def _get_last_human_message(messages: list) -> tuple[int, HumanMessage | None, str]:
    """Find the last human message and return its index, the message object, and content."""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return i, msg, content
            elif isinstance(content, list):
                # Handle multimodal content
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                return i, msg, " ".join(text_parts)
    return -1, None, ""


class SkillInjectionMiddleware(AgentMiddleware[Any, Any]):
    """Injects relevant skill content into the user message.
    
    This middleware analyzes the user's query and, if a relevant skill is found,
    appends its content to the user's message.
    
    IMPORTANT: We append to user message (not add SystemMessage) to ensure
    the system prompt remains completely static for prompt caching.
    """
    
    tools = []  # Required by middleware interface but not used
    
    # Backtest skills superseded by tool_backtest_strategy
    _BACKTEST_SKILL_IDS = frozenset({
        "backtest_ma_crossover",
        "backtest_macd",
        "backtest_bollinger",
        "backtest_chandelier_exit",
        "backtest_momentum_rotation",
    })
    
    def __init__(self, max_chars: int = MAX_SKILL_INJECT_CHARS, min_score: int = MIN_SKILL_SCORE):
        self.max_chars = max_chars
        self.min_score = min_score
    
    def _build_skill_suffix(self, skill_info: dict) -> str:
        """Build the skill content suffix to append to user message."""
        skill_id = skill_info.get("skill_id", "unknown")
        skill_name = skill_info.get("skill_name", skill_id)
        
        # Redirect backtest skills to tool_backtest_strategy
        if skill_id in self._BACKTEST_SKILL_IDS:
            return self._build_backtest_redirect(skill_id)
        
        skill_content = skill_info.get("content", "")
        
        return f"""

---
[Auto-Injected Skill: {skill_name}]

The system detected this query may benefit from the following skill patterns.
Follow these code patterns when writing Python:

{skill_content}

IMPORTANT: Before writing Python code, call `tool_search_and_load_skill("{skill_id}")` to:
1. Show users the skill you're using (builds trust)
2. Confirm you have the full guidance
---"""
    
    @staticmethod
    def _build_backtest_redirect(skill_id: str) -> str:
        """Instead of injecting a backtest skill, redirect to tool_backtest_strategy."""
        strategy_map = {
            "backtest_ma_crossover": "dual_ma",
            "backtest_macd": "macd",
            "backtest_bollinger": "bollinger",
            "backtest_chandelier_exit": "chandelier",
            "backtest_momentum_rotation": "momentum",
        }
        strategy = strategy_map.get(skill_id, "dual_ma")
        return f"""

---
[Auto-Redirect: Use tool_backtest_strategy instead of Python]

⚠️ Do NOT load skill "{skill_id}" or write Python backtest code.
Use `tool_backtest_strategy` — it handles the entire backtest pipeline in one call:

  tool_backtest_strategy(ts_codes=["<ts_code>"], strategy="{strategy}")

Supported strategies: dual_ma, bollinger, macd, chandelier, buy_and_hold, momentum.
The tool returns metrics (CAGR, Sharpe, MaxDD, etc.) + equity-curve chart automatically.
---"""
    
    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]) -> Any:
        """Inject skill content by appending to the user message."""
        messages = list(request.messages or [])
        
        # Find the last human message
        idx, human_msg, query = _get_last_human_message(messages)
        if idx < 0 or human_msg is None or not query.strip():
            return handler(request)
        
        # Check if skill content is already appended (avoid duplicate)
        if "[Skill Reference:" in query:
            return handler(request)
        
        # Select the most relevant skill
        skill_info = select_top_skill_for_query(
            query,
            max_content_chars=self.max_chars,
        )
        
        if skill_info is None or skill_info.get("score", 0) < self.min_score:
            return handler(request)
        
        # Append skill content to the user message
        skill_suffix = self._build_skill_suffix(skill_info)
        new_content = query + skill_suffix
        
        # Create new HumanMessage with appended content
        new_human_msg = HumanMessage(content=new_content)
        
        # Replace the original message
        new_messages = messages[:idx] + [new_human_msg] + messages[idx + 1:]
        
        # Create new request with modified messages
        new_request = ModelRequest(
            messages=new_messages,
            model=request.model,
            tools=request.tools,
            response_format=getattr(request, "response_format", None),
            state=getattr(request, "state", None),
        )
        
        return handler(new_request)
    
    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]) -> Any:
        """Async version of wrap_model_call."""
        messages = list(request.messages or [])
        
        # Find the last human message
        idx, human_msg, query = _get_last_human_message(messages)
        if idx < 0 or human_msg is None or not query.strip():
            return await handler(request)
        
        # Check if skill content is already appended
        if "[Skill Reference:" in query:
            return await handler(request)
        
        # Select the most relevant skill (run in thread to avoid blocking)
        skill_info = await asyncio.to_thread(
            select_top_skill_for_query,
            query,
            max_content_chars=self.max_chars,
        )
        
        if skill_info is None or skill_info.get("score", 0) < self.min_score:
            return await handler(request)
        
        # Append skill content to the user message
        skill_suffix = self._build_skill_suffix(skill_info)
        new_content = query + skill_suffix
        
        # Create new HumanMessage with appended content
        new_human_msg = HumanMessage(content=new_content)
        
        # Replace the original message
        new_messages = messages[:idx] + [new_human_msg] + messages[idx + 1:]
        
        # Create new request with modified messages
        new_request = ModelRequest(
            messages=new_messages,
            model=request.model,
            tools=request.tools,
            response_format=getattr(request, "response_format", None),
            state=getattr(request, "state", None),
        )
        
        return await handler(new_request)
