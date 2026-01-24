"""Token usage extraction and cost estimation utilities.

Provides dependency-free helpers to:
- Extract normalized usage from LangChain AIMessage objects
- Compute estimated cost using a JSON pricing config
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _default_pricing_path() -> Path:
    """Return path to the repo's pricing.json file."""
    # This file is: a-share-agent/src/agent/usage_cost.py
    # parents[0]=agent, [1]=src, [2]=a-share-agent
    return Path(__file__).resolve().parents[2] / "pricing.json"


_PRICING_CACHE: dict[str, Any] | None = None


def load_pricing(path: Path | str | None = None) -> dict[str, Any]:
    """Load pricing config from JSON file (cached after first load)."""
    global _PRICING_CACHE
    if _PRICING_CACHE is not None:
        return _PRICING_CACHE

    if path is None:
        env_path = os.environ.get("AGENT_PRICING_PATH")
        path = Path(env_path) if env_path else _default_pricing_path()
    else:
        path = Path(path)

    if not path.exists():
        # Return empty pricing if file missing (cost will be None)
        return {"models": {}, "default_model": None}

    with path.open("r", encoding="utf-8") as f:
        _PRICING_CACHE = json.load(f)

    return _PRICING_CACHE


def extract_usage(ai_msg: Any) -> dict[str, Any] | None:
    """Extract normalized usage dict from an AIMessage.

    Checks in order:
    1. ai_msg.usage_metadata (LangChain standard)
    2. ai_msg.response_metadata["token_usage"] (provider-specific)

    Returns a dict with keys:
    - input_tokens: int
    - output_tokens: int
    - total_tokens: int
    - input_cache_hit_tokens: int | None (DeepSeek specific)
    - input_cache_miss_tokens: int | None (DeepSeek specific)
    - reasoning_tokens: int | None (DeepSeek reasoner specific)
    - model: str | None

    Returns None if no usage info found.
    """
    if ai_msg is None:
        return None

    usage: dict[str, Any] = {}

    # Try usage_metadata first (LangChain standard)
    um = getattr(ai_msg, "usage_metadata", None)
    if um and isinstance(um, dict):
        usage["input_tokens"] = um.get("input_tokens", 0)
        usage["output_tokens"] = um.get("output_tokens", 0)
        usage["total_tokens"] = um.get("total_tokens", 0)
        # Check for cache details in input_token_details
        itd = um.get("input_token_details") or {}
        if isinstance(itd, dict):
            usage["input_cache_hit_tokens"] = itd.get("cache_read")
            usage["input_cache_miss_tokens"] = itd.get("cache_creation")
        # Check for reasoning in output_token_details
        otd = um.get("output_token_details") or {}
        if isinstance(otd, dict):
            usage["reasoning_tokens"] = otd.get("reasoning")

    # Also check response_metadata.token_usage (DeepSeek direct)
    rm = getattr(ai_msg, "response_metadata", None)
    if rm and isinstance(rm, dict):
        tu = rm.get("token_usage") or rm.get("usage") or {}
        if isinstance(tu, dict):
            # Merge/override with more complete info if present
            if tu.get("prompt_tokens"):
                usage["input_tokens"] = tu["prompt_tokens"]
            if tu.get("completion_tokens"):
                usage["output_tokens"] = tu["completion_tokens"]
            if tu.get("total_tokens"):
                usage["total_tokens"] = tu["total_tokens"]
            # DeepSeek cache fields
            if tu.get("prompt_cache_hit_tokens") is not None:
                usage["input_cache_hit_tokens"] = tu["prompt_cache_hit_tokens"]
            if tu.get("prompt_cache_miss_tokens") is not None:
                usage["input_cache_miss_tokens"] = tu["prompt_cache_miss_tokens"]
            # Reasoning tokens
            ctd = tu.get("completion_tokens_details") or {}
            if isinstance(ctd, dict) and ctd.get("reasoning_tokens") is not None:
                usage["reasoning_tokens"] = ctd["reasoning_tokens"]

        # Extract model name
        model_name = rm.get("model_name") or rm.get("model")
        if model_name:
            usage["model"] = model_name

    # Ensure we have at least input/output/total
    if not usage.get("input_tokens") and not usage.get("output_tokens"):
        return None

    # Ensure total_tokens is set
    if not usage.get("total_tokens"):
        usage["total_tokens"] = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

    return usage


def estimate_cost(
    usage: dict[str, Any],
    model_name: str | None = None,
    pricing: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Estimate cost in USD (or configured currency) for given usage.

    Args:
        usage: Normalized usage dict from extract_usage()
        model_name: Model identifier (e.g., "deepseek-chat"). Falls back to usage["model"] or pricing default.
        pricing: Pricing config dict. If None, loads from default path.

    Returns:
        {
            "currency": "USD",
            "total": 0.00123,
            "breakdown": {
                "input": 0.00100,
                "input_cache_hit": 0.00010,
                "input_cache_miss": 0.00090,
                "output": 0.00023
            }
        }
        Returns None if pricing not available for the model.
    """
    if not usage:
        return None

    if pricing is None:
        pricing = load_pricing()

    # Resolve model name
    model = model_name or usage.get("model") or pricing.get("default_model")
    if not model:
        return None

    # Find pricing for this model
    models = pricing.get("models", {})
    model_pricing = models.get(model)
    if not model_pricing:
        # Try partial match (e.g., "deepseek-chat" matches "deepseek-chat-v2")
        for k, v in models.items():
            if k in model or model in k:
                model_pricing = v
                break

    if not model_pricing:
        return None

    per_m = model_pricing.get("per_million_tokens", {})
    currency = model_pricing.get("currency", "USD")

    # Calculate input cost
    input_tokens = usage.get("input_tokens", 0)
    cache_hit = usage.get("input_cache_hit_tokens")
    cache_miss = usage.get("input_cache_miss_tokens")

    input_cost = 0.0
    input_cache_hit_cost = 0.0
    input_cache_miss_cost = 0.0

    if cache_hit is not None and cache_miss is not None:
        # Use detailed cache breakdown
        input_cache_hit_cost = (cache_hit / 1_000_000) * per_m.get("input_cache_hit", 0)
        input_cache_miss_cost = (cache_miss / 1_000_000) * per_m.get("input_cache_miss", 0)
        input_cost = input_cache_hit_cost + input_cache_miss_cost
    else:
        # Conservative: treat all input as cache miss
        input_cost = (input_tokens / 1_000_000) * per_m.get("input_cache_miss", 0)
        input_cache_miss_cost = input_cost

    # Calculate output cost
    output_tokens = usage.get("output_tokens", 0)
    output_cost = (output_tokens / 1_000_000) * per_m.get("output", 0)

    total_cost = input_cost + output_cost

    return {
        "currency": currency,
        "total": round(total_cost, 6),
        "breakdown": {
            "input": round(input_cost, 6),
            "input_cache_hit": round(input_cache_hit_cost, 6),
            "input_cache_miss": round(input_cache_miss_cost, 6),
            "output": round(output_cost, 6),
        },
    }


def compute_usage_and_cost(ai_msg: Any, model_name: str | None = None) -> dict[str, Any]:
    """Convenience: extract usage and estimate cost in one call.

    Returns:
        {
            "usage": {...} | None,
            "cost": {...} | None
        }
    """
    usage = extract_usage(ai_msg)
    cost = estimate_cost(usage, model_name=model_name) if usage else None
    return {"usage": usage, "cost": cost}
