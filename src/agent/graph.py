"""LangGraph agent definition for A-Share financial analysis."""

from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek

from agent.prompts import SYSTEM_PROMPT
from agent.tools import ALL_TOOLS

# Initialize the DeepSeek model
model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    temperature=0.1,  # Lower temperature for more precise financial analysis
)

# Create the agent graph using the new LangGraph 1.0 API
graph = create_agent(
    model=model,
    tools=ALL_TOOLS,
    system_prompt=SYSTEM_PROMPT,
)
