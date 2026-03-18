"""Structured per-user profile: portfolio, preferences, watchlist.

Design principles:
- **Structured, not free-text**: Pydantic models guarantee schema consistency.
  Portfolio data (ts_codes, cost bases, shares) is NEVER passed through mem0's
  lossy LLM-based fact extraction.
- **JSON persistence**: One JSON file per user_id on disk. Fast load, no
  embedding search needed — the profile is always fully injected at conversation
  start.
- **Dual-layer memory**: This module handles *structured* data (portfolio,
  preferences, watchlist). mem0 continues to handle *soft* memories (opinions,
  past Q&A, conversational facts).
- **Diff tracking**: Every portfolio update stores a timestamped snapshot so the
  agent can say "since last time, you sold X and bought Y".
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class Holding(BaseModel):
    """A single position in the user's portfolio."""
    name: str = Field(..., description="Display name, e.g. 贵州茅台")
    ts_code: str = Field("", description="Resolved tushare code, e.g. 600519.SH (empty if unresolved)")
    asset_type: str = Field("stock", description="stock | etf | index_etf | qdii | bond_fund | other")
    shares: float = Field(0, description="Number of shares/units held")
    cost_price: float = Field(0, description="Average cost price per share")
    current_price: float = Field(0, description="Latest known price")
    market_value: float = Field(0, description="Current market value (元)")
    pnl: float = Field(0, description="Unrealised P&L (元)")
    pnl_pct: float = Field(0, description="Unrealised P&L percentage, e.g. -0.15 = -15%")
    tags: list[str] = Field(default_factory=list, description="User-defined tags, e.g. ['美股科技', 'QDII']")


class WatchlistItem(BaseModel):
    """An asset the user is watching but not holding."""
    name: str
    ts_code: str = ""
    asset_type: str = "stock"
    reason: str = ""
    added_date: str = ""


class ActiveStrategy(BaseModel):
    """A strategy the user has expressed interest in or is running."""
    name: str = Field(..., description="Strategy name, e.g. dual_moving_average")
    description: str = ""
    ts_codes: list[str] = Field(default_factory=list, description="Applicable ts_codes")
    params: dict = Field(default_factory=dict, description="Strategy parameters")
    notes: str = ""


class UserPreferences(BaseModel):
    """Investment preferences and style."""
    risk_tolerance: str = Field("moderate", description="conservative | moderate | aggressive")
    investment_horizon: str = Field("medium", description="short (<6mo) | medium (6mo-3yr) | long (>3yr)")
    preferred_sectors: list[str] = Field(default_factory=list, description="e.g. ['科技', '消费', '医药']")
    avoided_sectors: list[str] = Field(default_factory=list)
    rebalance_frequency: str = Field("", description="e.g. monthly, quarterly")
    max_single_position_pct: float = Field(0.25, description="Max weight for single position, 0-1")
    target_cash_pct: float = Field(0.05, description="Target cash allocation, 0-1")
    notes: str = Field("", description="Free-text notes about user preferences")


class PortfolioSnapshot(BaseModel):
    """A timestamped copy of the portfolio for diff tracking."""
    timestamp: str = Field(default_factory=lambda: _now_iso())
    total_assets: float = 0
    total_market_value: float = 0
    cash: float = 0
    holdings: list[Holding] = Field(default_factory=list)


class UserProfile(BaseModel):
    """Root model — one per user. Persisted as JSON."""
    user_id: str
    created_at: str = Field(default_factory=lambda: _now_iso())
    updated_at: str = Field(default_factory=lambda: _now_iso())

    # --- Portfolio ---
    total_assets: float = 0
    total_market_value: float = 0
    cash: float = 0
    holdings: list[Holding] = Field(default_factory=list)

    # --- Preferences & strategies ---
    preferences: UserPreferences = Field(default_factory=lambda: UserPreferences())  # type: ignore[call-arg]
    strategies: list[ActiveStrategy] = Field(default_factory=list)
    watchlist: list[WatchlistItem] = Field(default_factory=list)

    # --- History (last N snapshots for diff tracking) ---
    snapshots: list[PortfolioSnapshot] = Field(default_factory=list)

    # --- Misc ---
    custom_data: dict = Field(default_factory=dict, description="Extensible key-value store")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BJ_TZ = timezone(timedelta(hours=8))


def _now_iso() -> str:
    return datetime.now(_BJ_TZ).isoformat(timespec="seconds")


# Default storage directory — sibling to mem0_qdrant
_DEFAULT_PROFILE_DIR = Path(__file__).resolve().parents[3] / "data" / "user_profiles"


_profile_dir_ensured: str | None = None  # path that was last mkdir'd


def _profile_dir() -> Path:
    global _profile_dir_ensured
    d = Path(os.environ.get("USER_PROFILE_DIR", str(_DEFAULT_PROFILE_DIR)))
    d_str = str(d)
    if _profile_dir_ensured != d_str:
        d.mkdir(parents=True, exist_ok=True)
        _profile_dir_ensured = d_str
    return d


def _profile_path(user_id: str) -> Path:
    # Sanitise user_id for filesystem
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
    return _profile_dir() / f"{safe}.json"


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def load_profile(user_id: str) -> Optional[UserProfile]:
    """Load a user profile from disk. Returns None if not found."""
    path = _profile_path(user_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return UserProfile.model_validate(data)
    except Exception as exc:
        print(f"[user_profile] Failed to load {path}: {exc}")
        return None


def save_profile(profile: UserProfile) -> Path:
    """Persist a user profile to disk atomically. Returns the file path."""
    profile.updated_at = _now_iso()
    path = _profile_path(profile.user_id)
    data = profile.model_dump_json(indent=2)
    # Atomic write: write to temp file in same dir, then rename
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=".profile_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        # os.replace is atomic on both POSIX and Windows (same volume)
        os.replace(tmp_path, str(path))
    except BaseException:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return path


def get_or_create_profile(user_id: str) -> UserProfile:
    """Load existing profile or create a blank one."""
    profile = load_profile(user_id)
    if profile is None:
        profile = UserProfile(user_id=user_id)
        save_profile(profile)
    return profile


# ---------------------------------------------------------------------------
# Portfolio update with diff tracking
# ---------------------------------------------------------------------------

_MAX_SNAPSHOTS = 20  # Keep last N snapshots


def update_portfolio(
    user_id: str,
    *,
    holdings: list[dict],
    total_assets: float = 0,
    total_market_value: float = 0,
    cash: float = 0,
    mode: str = "replace",
) -> dict:
    """Update the user's portfolio, storing a snapshot of the old one.

    Args:
        user_id: The user identifier.
        holdings: List of holding dicts (will be validated as Holding models).
        total_assets: Total account assets.
        total_market_value: Sum of all position market values.
        cash: Available cash.
        mode: "replace" (default) overwrites all holdings.
              "merge" adds/updates the given holdings while keeping
              existing ones that are not mentioned.

    Returns:
        {saved: bool, diff: {...}, profile_summary: str}
    """
    profile = get_or_create_profile(user_id)

    # Snapshot old state before overwriting
    if profile.holdings:
        old_snapshot = PortfolioSnapshot(
            total_assets=profile.total_assets,
            total_market_value=profile.total_market_value,
            cash=profile.cash,
            holdings=[h.model_copy() for h in profile.holdings],
        )
        profile.snapshots.append(old_snapshot)
        # Trim to last N
        if len(profile.snapshots) > _MAX_SNAPSHOTS:
            profile.snapshots = profile.snapshots[-_MAX_SNAPSHOTS:]

    # Validate new holdings individually
    old_codes = {h.ts_code or h.name for h in profile.holdings}
    incoming: list[Holding] = []
    validation_errors: list[dict] = []
    for i, h in enumerate(holdings):
        try:
            incoming.append(Holding.model_validate(h))
        except Exception as exc:
            name = h.get("name", f"holding[{i}]") if isinstance(h, dict) else f"holding[{i}]"
            validation_errors.append({"index": i, "name": name, "error": str(exc)})
            logger.warning("[user_profile] Skipped invalid holding %s: %s", name, exc)

    if mode == "merge":
        # Build a lookup of existing holdings keyed by (ts_code or name)
        existing_map = {(h.ts_code or h.name): h for h in profile.holdings}
        for h in incoming:
            existing_map[h.ts_code or h.name] = h  # add or overwrite
        new_holdings = list(existing_map.values())
    else:
        new_holdings = incoming

    new_codes = {h.ts_code or h.name for h in new_holdings}
    added = new_codes - old_codes
    removed = old_codes - new_codes
    kept = old_codes & new_codes

    # Update
    profile.holdings = new_holdings
    if total_assets:
        profile.total_assets = total_assets
    if total_market_value:
        profile.total_market_value = total_market_value
    if cash:
        profile.cash = cash

    save_profile(profile)

    result: dict = {
        "saved": True,
        "mode": mode,
        "diff": {
            "added": sorted(added),
            "removed": sorted(removed),
            "kept_count": len(kept),
            "old_snapshot_stored": True,
        },
        "profile_summary": format_portfolio_summary(profile),
    }
    if validation_errors:
        result["validation_errors"] = validation_errors
        result["skipped_count"] = len(validation_errors)
    return result


def update_preferences(user_id: str, **kwargs) -> dict:
    """Update user preferences. Accepts any field of UserPreferences."""
    profile = get_or_create_profile(user_id)
    for k, v in kwargs.items():
        if hasattr(profile.preferences, k):
            setattr(profile.preferences, k, v)
    save_profile(profile)
    return {"saved": True, "preferences": profile.preferences.model_dump()}


def add_watchlist_item(user_id: str, item: dict) -> dict:
    """Add an item to the user's watchlist."""
    profile = get_or_create_profile(user_id)
    wi = WatchlistItem.model_validate(item)
    # Avoid duplicates by ts_code or name
    existing = {(w.ts_code or w.name) for w in profile.watchlist}
    key = wi.ts_code or wi.name
    if key in existing:
        return {"added": False, "reason": f"'{key}' already in watchlist"}
    wi.added_date = _now_iso()
    profile.watchlist.append(wi)
    save_profile(profile)
    return {"added": True, "watchlist_count": len(profile.watchlist)}


def remove_watchlist_item(user_id: str, name_or_code: str) -> dict:
    """Remove an item from the watchlist by name or ts_code."""
    profile = get_or_create_profile(user_id)
    before = len(profile.watchlist)
    profile.watchlist = [
        w for w in profile.watchlist
        if w.ts_code != name_or_code and w.name != name_or_code
    ]
    removed = before - len(profile.watchlist)
    if removed:
        save_profile(profile)
    return {"removed": removed, "watchlist_count": len(profile.watchlist)}


def add_strategy(user_id: str, strategy: dict) -> dict:
    """Add or update an active strategy."""
    profile = get_or_create_profile(user_id)
    strat = ActiveStrategy.model_validate(strategy)
    # Replace if same name exists
    profile.strategies = [s for s in profile.strategies if s.name != strat.name]
    profile.strategies.append(strat)
    save_profile(profile)
    return {"saved": True, "strategy_count": len(profile.strategies)}


# ---------------------------------------------------------------------------
# Formatting helpers (for injection into system message)
# ---------------------------------------------------------------------------

def format_portfolio_summary(profile: UserProfile) -> str:
    """Format the portfolio as a concise text block for LLM context injection."""
    if not profile.holdings:
        return "(No portfolio data recorded yet)"

    lines = [
        f"## 📊 User Portfolio (updated {profile.updated_at})",
        f"Total Assets: ¥{profile.total_assets:,.2f}",
        f"Market Value: ¥{profile.total_market_value:,.2f}",
        f"Cash: ¥{profile.cash:,.2f}",
        f"Position Ratio: {profile.total_market_value / profile.total_assets * 100:.1f}%"
        if profile.total_assets > 0 else "",
        "",
        "| # | Name | Code | Shares | Cost | Price | Value | P&L | P&L% | Tags |",
        "|---|------|------|--------|------|-------|-------|-----|------|------|",
    ]
    for i, h in enumerate(profile.holdings, 1):
        pnl_str = f"{h.pnl:+,.0f}" if h.pnl else "0"
        pnl_pct_str = f"{h.pnl_pct * 100:+.2f}%" if h.pnl_pct else "0%"
        tags_str = ", ".join(h.tags) if h.tags else ""
        lines.append(
            f"| {i} | {h.name} | {h.ts_code} | {h.shares:,.0f} "
            f"| {h.cost_price:.3f} | {h.current_price:.3f} "
            f"| ¥{h.market_value:,.0f} | {pnl_str} | {pnl_pct_str} | {tags_str} |"
        )
    return "\n".join(lines)


def format_preferences_summary(profile: UserProfile) -> str:
    """Format preferences as a text block."""
    p = profile.preferences
    parts = [
        f"Risk: {p.risk_tolerance}",
        f"Horizon: {p.investment_horizon}",
    ]
    if p.preferred_sectors:
        parts.append(f"Preferred: {', '.join(p.preferred_sectors)}")
    if p.avoided_sectors:
        parts.append(f"Avoided: {', '.join(p.avoided_sectors)}")
    if p.notes:
        parts.append(f"Notes: {p.notes}")
    return " | ".join(parts)


def format_watchlist_summary(profile: UserProfile) -> str:
    """Format watchlist as a text block."""
    if not profile.watchlist:
        return "(Empty watchlist)"
    lines = []
    for w in profile.watchlist:
        reason = f" — {w.reason}" if w.reason else ""
        lines.append(f"- {w.name} ({w.ts_code}){reason}")
    return "\n".join(lines)


def format_strategies_summary(profile: UserProfile) -> str:
    """Format active strategies."""
    if not profile.strategies:
        return "(No active strategies)"
    lines = []
    for s in profile.strategies:
        codes = ", ".join(s.ts_codes) if s.ts_codes else "N/A"
        lines.append(f"- **{s.name}**: {s.description} [codes: {codes}]")
    return "\n".join(lines)


def format_full_profile_context(profile: UserProfile) -> str:
    """Format the entire profile as a context block for LLM injection.

    This is what gets prepended to the system message at conversation start.
    """
    sections = [
        format_portfolio_summary(profile),
    ]

    prefs = format_preferences_summary(profile)
    if prefs:
        sections.append(f"\n**Preferences:** {prefs}")

    wl = format_watchlist_summary(profile)
    if wl and wl != "(Empty watchlist)":
        sections.append(f"\n**Watchlist:**\n{wl}")

    strats = format_strategies_summary(profile)
    if strats and strats != "(No active strategies)":
        sections.append(f"\n**Active Strategies:**\n{strats}")

    if profile.snapshots:
        last = profile.snapshots[-1]
        sections.append(
            f"\n**Previous snapshot** ({last.timestamp}): "
            f"Assets ¥{last.total_assets:,.0f}, "
            f"{len(last.holdings)} holdings"
        )

    return "\n".join(sections)
