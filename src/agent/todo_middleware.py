"""Smart todo middleware tailored for A-Share investment advisor.

The stock TodoListMiddleware is designed for generic code agents.  We keep
the same state schema (``PlanningState``) and tool name (``write_todos``)
so the existing chat-ui renders the todo list unchanged, but tune the
system prompt and behaviour for a *personal investment advisor*:

Differences from the stock TodoListMiddleware
---------------------------------------------
1. **Investment-aware system prompt** – guides the LLM to plan in terms
   of research steps, data gathering, cross-validation, risk checks.
2. **Fewer unnecessary calls** – the prompt explicitly tells the model
   NOT to use todos for simple price lookups or single-stock queries,
   only for multi-asset analysis, portfolio reviews, strategy work.
3. **Cancelled state** – the front-end already supports a ``cancelled``
   status; we surface it in the prompt so the model can drop irrelevant
   steps instead of completing them with bogus results.
4. **No change to tool schema** – ``write_todos`` still accepts
   ``list[Todo]`` and updates ``state["todos"]``, keeping full
   backwards-compatibility with the chat-ui ``TodoListCard`` component.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.runtime import Runtime

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    OmitFromInput,
)
from langchain.tools import InjectedToolCallId


# ---------------------------------------------------------------------------
# State schema – identical to stock TodoListMiddleware so LangGraph merges it
# ---------------------------------------------------------------------------

class Todo(TypedDict):
    """A single todo item with content and status."""

    content: str
    """The content/description of the todo item."""

    status: Literal["pending", "in_progress", "completed", "cancelled"]
    """The current status of the todo item."""


class PlanningState(AgentState[Any]):
    """State schema for the todo middleware."""

    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]
    """List of todo items for tracking task progress."""


# ---------------------------------------------------------------------------
# Prompt & description – investment-advisor flavour
# ---------------------------------------------------------------------------

WRITE_TODOS_TOOL_DESCRIPTION = """\
Manage a structured research/analysis plan for the current session.
Use this tool to give the user visibility into your multi-step work.

## When to use
- Portfolio review / 向我汇报 (multiple holdings to check)
- Multi-asset comparison or industry analysis (≥3 assets)
- Strategy backtesting or evaluation involving several steps
- Complex investment questions requiring data gathering → analysis → recommendation
- User explicitly asks you to plan or lists multiple tasks

## When NOT to use (just do the work directly)
- Single stock lookup or price check (1-2 tool calls)
- Simple factual questions (one tool call answers it)
- Conversational follow-ups within an ongoing analysis
- Anything that takes fewer than 3 steps

## Task states
- **pending** – not yet started
- **in_progress** – currently working on
- **completed** – fully finished
- **cancelled** – no longer relevant (e.g. data unavailable, user changed scope)

## Rules
- Mark a task **in_progress** BEFORE you start it.
- Mark it **completed** IMMEDIATELY after finishing — do not batch.
- If data is unavailable or a step becomes irrelevant, mark it **cancelled** with
  a brief note in ``content`` explaining why.
- Keep the list concise: 3-6 items. Avoid micro-steps like "resolve ts_code".
- You may revise the plan as new information emerges.
- NEVER call write_todos multiple times in parallel.
"""

WRITE_TODOS_SYSTEM_PROMPT = """\
## 📋 Research Plan (`write_todos`)

You have a `write_todos` tool to organise multi-step research for the user.

**Use it when** the task has ≥3 meaningful steps (portfolio review, multi-asset \
comparison, strategy evaluation). **Skip it for** simple lookups (single stock \
price, one indicator, quick factual answer).

When you do use it:
- Create a concise plan (3-6 steps) focusing on *what the user needs*, \
not internal plumbing.
- Mark steps in_progress → completed as you go.
- If a step becomes impossible (data missing, API error), mark it **cancelled** \
and explain briefly.
- Do NOT call write_todos in parallel with itself.
"""


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------

def _make_write_todos_tool(description: str):
    """Create the write_todos tool with the given description."""

    @tool(description=description)
    def write_todos(
        todos: list[Todo],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command[Any]:
        """Create and manage a structured task list for your current work session."""
        return Command(
            update={
                "todos": todos,
                "messages": [
                    ToolMessage(
                        f"Updated todo list to {todos}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return write_todos


# ---------------------------------------------------------------------------
# Middleware class
# ---------------------------------------------------------------------------

class InvestmentTodoMiddleware(AgentMiddleware):
    """Todo middleware tuned for a personal investment advisor.

    Drop-in replacement for ``TodoListMiddleware`` — same state schema,
    same ``write_todos`` tool name, same front-end contract.
    """

    state_schema = PlanningState

    def __init__(
        self,
        *,
        system_prompt: str = WRITE_TODOS_SYSTEM_PROMPT,
        tool_description: str = WRITE_TODOS_TOOL_DESCRIPTION,
    ) -> None:
        super().__init__()
        self.system_prompt = system_prompt
        self.tool_description = tool_description
        self.tools = [_make_write_todos_tool(self.tool_description)]

    # -- model call: inject system prompt --------------------------------

    def _inject_system(self, request: ModelRequest) -> ModelRequest:
        if request.system_message is not None:
            new_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_content = [{"type": "text", "text": self.system_prompt}]
        new_msg = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_content),
        )
        return request.override(system_message=new_msg)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self._inject_system(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._inject_system(request))

    # -- after model: guard parallel write_todos calls -------------------

    @override
    def after_model(
        self, state: AgentState[Any], runtime: "Runtime"
    ) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not messages:
            return None

        last_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)), None
        )
        if not last_ai or not last_ai.tool_calls:
            return None

        wt_calls = [
            tc for tc in last_ai.tool_calls if tc["name"] == "write_todos"
        ]
        if len(wt_calls) > 1:
            return {
                "messages": [
                    ToolMessage(
                        content=(
                            "Error: `write_todos` must not be called multiple "
                            "times in parallel. Please call it once per turn."
                        ),
                        tool_call_id=tc["id"],
                        status="error",
                    )
                    for tc in wt_calls
                ]
            }
        return None

    @override
    async def aafter_model(
        self, state: AgentState[Any], runtime: "Runtime"
    ) -> dict[str, Any] | None:
        return self.after_model(state, runtime)
