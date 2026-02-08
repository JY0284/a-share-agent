"""LangGraph agent definition for A-Share financial analysis."""

from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_deepseek import ChatDeepSeek

from agent.prompts import get_system_prompt
from agent.python_guard_middleware import PythonGuardMiddleware
from agent.trace_middleware import LocalTraceMiddleware
from agent.tools import ALL_TOOLS
from agent.web_search import WEB_SEARCH_TOOLS, get_tavily_api_key

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
    return tools

# Create the agent graph using LangGraph 1.0 API
graph = create_agent(
    model=model,
    tools=get_all_tools(),
    system_prompt=get_system_prompt(),
    middleware=[LocalTraceMiddleware(), PythonGuardMiddleware(), TodoListMiddleware()],
)
