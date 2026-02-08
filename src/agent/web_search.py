"""Web search tools using Tavily API.

This module provides web search capabilities for the A-Share agent.
Web search can be controlled via:
1. Environment variable WEB_SEARCH_ENABLED (server-side default)
2. Request context web_search_enabled (per-request override from frontend)
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig


def is_web_search_enabled_by_env() -> bool:
    """Check if web search is enabled via environment variable (server default)."""
    return os.environ.get("WEB_SEARCH_ENABLED", "false").lower() in ("true", "1", "yes")


def is_web_search_enabled(config: Optional[RunnableConfig] = None) -> bool:
    """Check if web search is enabled.
    
    Priority:
    1. If config contains 'web_search_enabled' in configurable, use that
    2. Otherwise fall back to environment variable
    
    Args:
        config: Optional RunnableConfig that may contain web_search_enabled setting
        
    Returns:
        True if web search should be enabled for this request
    """
    # Check if there's a per-request override in the config
    if config:
        configurable = config.get("configurable", {})
        if "web_search_enabled" in configurable:
            return bool(configurable["web_search_enabled"])
    
    # Fall back to environment variable
    return is_web_search_enabled_by_env()


def get_tavily_api_key() -> Optional[str]:
    """Get Tavily API key from environment."""
    return os.environ.get("TAVILY_API_KEY")


def get_web_search_tools() -> list:
    """Get web search tools if configured.
    
    Note: This now always returns the tools if TAVILY_API_KEY is set.
    The actual enable/disable check happens at runtime in the tool itself.

    Returns:
        List of web search tools if Tavily is configured, empty list otherwise.
    """
    api_key = get_tavily_api_key()
    if not api_key:
        if is_web_search_enabled_by_env():
            print("Warning: WEB_SEARCH_ENABLED=true but TAVILY_API_KEY not set")
        return []

    try:
        from langchain_tavily import TavilySearch

        # Create the Tavily search tool
        tavily_search = TavilySearch(
            max_results=5,
            topic="general",
        )

        return [tavily_search]
    except ImportError:
        print("Warning: langchain-tavily not installed. Run: pip install langchain-tavily")
        return []
    except Exception as e:
        print(f"Warning: Failed to initialize Tavily search: {e}")
        return []


@tool
def tool_web_search(query: str, max_results: int = 3, *, config: Optional[RunnableConfig] = None) -> dict:
    """Search the web for real-time information using Tavily.

    Use this tool when you need to:
    - Get current/real-time information about news, events, or market updates
    - Find information not available in your local data store
    - Research topics beyond A-share stock data
    - Verify or supplement local data with external sources

    Note: This tool is for general web search. For A-share specific data,
    prefer using the local data tools (tool_get_daily_prices, etc.)

    Args:
        query: The search query in natural language
        max_results: Maximum number of results to return (default: 3, max: 10)

    Returns:
        {
            "results": [
                {
                    "title": "Page title",
                    "url": "Page URL",
                    "content": "Relevant excerpt from the page",
                    "score": 0.95  # Relevance score
                },
                ...
            ],
            "query": "Original query",
            "error": null  # Or error message if failed
        }
    """
    # Check if web search is enabled (checks config first, then env var)
    if not is_web_search_enabled(config):
        return {
            "results": [],
            "query": query,
            "error": "Web search is disabled. Enable it in settings or set WEB_SEARCH_ENABLED=true."
        }

    api_key = get_tavily_api_key()
    if not api_key:
        return {
            "results": [],
            "query": query,
            "error": "TAVILY_API_KEY not configured."
        }

    try:
        from langchain_tavily import TavilySearch

        # Limit max_results
        max_results = min(max(1, max_results), 10)

        tavily = TavilySearch(max_results=max_results)
        raw_results = tavily.invoke(query)

        # Parse and structure the results - Tavily returns nested structure
        results = []
        if isinstance(raw_results, dict):
            # Direct response with 'results' key
            if 'results' in raw_results:
                results = raw_results['results']
            else:
                results = [raw_results]
        elif isinstance(raw_results, list):
            # List of responses
            for item in raw_results:
                if isinstance(item, dict) and 'results' in item:
                    results.extend(item['results'])
                else:
                    results.append(item)
        elif isinstance(raw_results, str):
            results = [{"content": raw_results}]

        return {
            "results": results,
            "query": query,
            "error": None
        }

    except ImportError:
        return {
            "results": [],
            "query": query,
            "error": "langchain-tavily not installed."
        }
    except Exception as e:
        return {
            "results": [],
            "query": query,
            "error": str(e)
        }


# List of web search tools - always include if Tavily is configured
# The runtime check happens inside the tool
WEB_SEARCH_TOOLS = [tool_web_search]
