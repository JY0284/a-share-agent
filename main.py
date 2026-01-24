#!/usr/bin/env python3
"""Entry point for testing the A-Share agent locally."""

from __future__ import annotations

import os
import sys

# Ensure the src directory is in the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Load environment variables from .env file
from pathlib import Path

env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def main():
    """Test the agent with a simple query."""
    from langchain_core.messages import HumanMessage

    from agent.graph import graph

    print("A-Share Agent Test")
    print("=" * 50)

    # Test query
    query = "查询一下贵州茅台(600519)的基本信息和最近5个交易日的价格"

    print(f"\nQuery: {query}\n")
    print("-" * 50)

    # Invoke the agent
    result = graph.invoke({"messages": [HumanMessage(content=query)]})

    # Print the final response
    for message in result["messages"]:
        if hasattr(message, "content") and message.content:
            print(f"\n[{message.__class__.__name__}]")
            print(message.content)


if __name__ == "__main__":
    main()
