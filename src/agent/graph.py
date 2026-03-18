"""LangGraph agent definition for A-Share financial analysis."""

from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek

from agent.prompts import get_system_prompt
from agent.python_guard_middleware import PythonGuardMiddleware
from agent.skill_injection_middleware import SkillInjectionMiddleware
from agent.trace_middleware import LocalTraceMiddleware
from agent.tools import ALL_TOOLS
from agent.web_search import WEB_SEARCH_TOOLS, get_tavily_api_key
from agent.memory import MEMORY_TOOLS, MEM0_ENABLED
from agent.memory_middleware import MemoryMiddleware
from agent.profile_tools import PROFILE_TOOLS
from agent.batch_tools import BATCH_TOOLS
from agent.todo_middleware import InvestmentTodoMiddleware

# Initialize the DeepSeek model
model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0.1,  # Lower temperature for more precise financial analysis
)

# Combine tools: always include ALL_TOOLS, add web search tools if Tavily is configured
def get_all_tools():
    tools = list(ALL_TOOLS)
    if get_tavily_api_key():
        tools.extend(WEB_SEARCH_TOOLS)
    # Add memory tools if mem0 is enabled
    if MEM0_ENABLED:
        tools.extend(MEMORY_TOOLS)
    # Always include profile + batch tools (no external dependency)
    tools.extend(PROFILE_TOOLS)
    tools.extend(BATCH_TOOLS)
    return tools

# Build middleware stack
def get_middleware():
    middleware = []
    # Memory middleware first: injects UserProfile + relevant soft memories
    middleware.append(MemoryMiddleware())
    middleware.extend([
        SkillInjectionMiddleware(),  # Inject relevant skill content based on query
        LocalTraceMiddleware(),      # Traces the MODIFIED messages
        PythonGuardMiddleware(),
        InvestmentTodoMiddleware(),
    ])
    return middleware

# Create the agent graph using LangGraph 1.0 API
graph = create_agent(
    model=model,
    tools=get_all_tools(),
    system_prompt=get_system_prompt(),
    middleware=get_middleware(),
)
