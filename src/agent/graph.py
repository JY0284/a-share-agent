"""LangGraph agent definition for A-Share financial analysis."""

from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_deepseek import ChatDeepSeek

from agent.prompts import get_system_prompt
from agent.tools import ALL_TOOLS

# Initialize the DeepSeek model
model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0.1,  # Lower temperature for more precise financial analysis
)

# Create the agent graph using LangGraph 1.0 API
# The system prompt is generated fresh at import time with current datetime
graph = create_agent(
    model=model,
    tools=ALL_TOOLS,
    system_prompt=get_system_prompt(),
    middleware=[TodoListMiddleware()]
)
