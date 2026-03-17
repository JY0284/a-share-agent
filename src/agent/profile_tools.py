"""LangChain tool wrappers for structured user profile operations.

These tools give the agent explicit, reliable CRUD over the user's:
- Portfolio (holdings, cash, total assets)
- Preferences (risk tolerance, horizon, sectors)
- Watchlist (assets being monitored)
- Strategies (active investment strategies)

Unlike mem0-based memory tools (which store free-text and may lose structure),
these tools persist data as validated Pydantic models in JSON files.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from agent.user_profile import (
    get_or_create_profile,
    update_portfolio,
    update_preferences,
    add_watchlist_item,
    remove_watchlist_item,
    add_strategy,
    format_portfolio_summary,
)


def _get_user_id(config: RunnableConfig) -> str:
    """Extract user_id from LangGraph runtime config."""
    configurable = config.get("configurable", {})
    uid = configurable.get("user_id")
    if uid:
        return str(uid)
    # Dev-mode fallback
    return "dev_user"


# ---------------------------------------------------------------------------
# Portfolio tools
# ---------------------------------------------------------------------------

@tool
def tool_get_portfolio(*, config: RunnableConfig) -> dict:
    """Get the user's current portfolio (holdings, cash, total assets).

    Returns the full structured portfolio. Use this FIRST when the user asks
    about their positions, wants a report (汇报), or asks "my portfolio".

    The profile also includes preferences, watchlist, and active strategies.

    Returns:
        {has_portfolio: bool, summary: str, holdings: [...], total_assets, cash, ...}
    """
    user_id = _get_user_id(config)
    profile = get_or_create_profile(user_id)

    if not profile.holdings:
        return {
            "has_portfolio": False,
            "summary": "No portfolio data recorded yet. Ask the user to share their holdings.",
            "hint": "When the user shares portfolio data, call tool_update_portfolio to save it.",
        }

    return {
        "has_portfolio": True,
        "summary": format_portfolio_summary(profile),
        "total_assets": profile.total_assets,
        "total_market_value": profile.total_market_value,
        "cash": profile.cash,
        "holdings": [h.model_dump() for h in profile.holdings],
        "preferences": profile.preferences.model_dump(),
        "watchlist": [w.model_dump() for w in profile.watchlist],
        "strategies": [s.model_dump() for s in profile.strategies],
        "updated_at": profile.updated_at,
        "snapshot_count": len(profile.snapshots),
    }


@tool
def tool_update_portfolio(
    holdings: list[dict],
    total_assets: float = 0,
    total_market_value: float = 0,
    cash: float = 0,
    *,
    config: RunnableConfig,
) -> dict:
    """Save/update the user's portfolio with structured holding data.

    Call this whenever the user shares their portfolio (持仓). Parse the user's
    data into the structured format before calling.

    Each holding dict should have:
    - name (str, required): Display name, e.g. "贵州茅台"
    - ts_code (str): Resolved code, e.g. "600519.SH" — resolve via tool_smart_search if needed
    - asset_type (str): "stock" | "etf" | "qdii" | "index_etf" | "other"
    - shares (float): Number of shares held
    - cost_price (float): Average cost per share
    - current_price (float): Latest price
    - market_value (float): Current market value in 元
    - pnl (float): Unrealised profit/loss in 元
    - pnl_pct (float): P&L as decimal, e.g. -0.15 means -15%
    - tags (list[str]): Classification tags, e.g. ["美股科技", "QDII"]

    IMPORTANT: Always try to resolve ts_code using tool_smart_search before saving.
    This ensures future batch price lookups work correctly.

    Args:
        holdings: List of holding dicts
        total_assets: Total account assets (总资产)
        total_market_value: Sum of position market values (总市值)
        cash: Available cash (可用资金)

    Returns:
        {saved: bool, diff: {added, removed, kept_count}, profile_summary: str}
    """
    user_id = _get_user_id(config)
    return update_portfolio(
        user_id,
        holdings=holdings,
        total_assets=total_assets,
        total_market_value=total_market_value,
        cash=cash,
    )


@tool
def tool_update_preferences(
    risk_tolerance: str = "",
    investment_horizon: str = "",
    preferred_sectors: list[str] | None = None,
    avoided_sectors: list[str] | None = None,
    max_single_position_pct: float = 0,
    target_cash_pct: float = 0,
    notes: str = "",
    *,
    config: RunnableConfig,
) -> dict:
    """Update the user's investment preferences.

    Call when the user states their risk appetite, time horizon, or sector
    preferences. Only non-empty/non-zero values are updated.

    Args:
        risk_tolerance: "conservative" | "moderate" | "aggressive"
        investment_horizon: "short" (<6mo) | "medium" (6mo-3yr) | "long" (>3yr)
        preferred_sectors: Sectors the user likes, e.g. ["科技", "消费"]
        avoided_sectors: Sectors to avoid
        max_single_position_pct: Max weight for a single position (0-1)
        target_cash_pct: Target cash allocation (0-1)
        notes: Free-text notes
    """
    user_id = _get_user_id(config)
    kwargs = {}
    if risk_tolerance:
        kwargs["risk_tolerance"] = risk_tolerance
    if investment_horizon:
        kwargs["investment_horizon"] = investment_horizon
    if preferred_sectors is not None:
        kwargs["preferred_sectors"] = preferred_sectors
    if avoided_sectors is not None:
        kwargs["avoided_sectors"] = avoided_sectors
    if max_single_position_pct > 0:
        kwargs["max_single_position_pct"] = max_single_position_pct
    if target_cash_pct > 0:
        kwargs["target_cash_pct"] = target_cash_pct
    if notes:
        kwargs["notes"] = notes
    if not kwargs:
        return {"saved": False, "reason": "No preference values provided"}
    return update_preferences(user_id, **kwargs)


@tool
def tool_add_watchlist(
    name: str,
    ts_code: str = "",
    asset_type: str = "stock",
    reason: str = "",
    *,
    config: RunnableConfig,
) -> dict:
    """Add an asset to the user's watchlist.

    Call when the user says "help me watch X" or "关注一下XX".

    Args:
        name: Asset display name
        ts_code: Resolved ts_code (resolve first via tool_smart_search)
        asset_type: "stock" | "etf" | "index" | etc.
        reason: Why the user is watching this
    """
    user_id = _get_user_id(config)
    return add_watchlist_item(user_id, {
        "name": name,
        "ts_code": ts_code,
        "asset_type": asset_type,
        "reason": reason,
    })


@tool
def tool_remove_watchlist(
    name_or_code: str,
    *,
    config: RunnableConfig,
) -> dict:
    """Remove an asset from the user's watchlist.

    Args:
        name_or_code: Name or ts_code of the item to remove
    """
    user_id = _get_user_id(config)
    return remove_watchlist_item(user_id, name_or_code)


@tool
def tool_add_strategy(
    name: str,
    description: str = "",
    ts_codes: list[str] | None = None,
    params: dict | None = None,
    notes: str = "",
    *,
    config: RunnableConfig,
) -> dict:
    """Record an active investment strategy for the user.

    Call when the user sets up or discusses a specific strategy (e.g.
    "I want to run dual MA on 茅台" or "定投纳斯达克ETF every month").

    Args:
        name: Strategy name (e.g. "dual_moving_average", "定投", "target_weight")
        description: Brief description
        ts_codes: Which assets this strategy applies to
        params: Strategy parameters (e.g. {"fast_period": 5, "slow_period": 20})
        notes: Additional context
    """
    user_id = _get_user_id(config)
    return add_strategy(user_id, {
        "name": name,
        "description": description,
        "ts_codes": ts_codes or [],
        "params": params or {},
        "notes": notes,
    })


# ---------------------------------------------------------------------------
# Exported list
# ---------------------------------------------------------------------------

PROFILE_TOOLS = [
    tool_update_portfolio,
    tool_update_preferences,
    tool_add_watchlist,
    tool_remove_watchlist,
    tool_add_strategy,
]
