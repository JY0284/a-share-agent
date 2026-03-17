"""Daily briefing routine — generates a personalised morning report.

This module is designed to be called either:
  1. As a scheduled cron job (via `run_daily_briefing(user_id)`)
  2. As an on-demand routine invoked by the agent when the user says "向我汇报"

The routine orchestrates existing tools programmatically (no LLM needed for
data gathering) and returns a structured dict that the agent can format into
a natural-language report.

Architecture:
  run_daily_briefing(user_id)
    ├─ load UserProfile
    ├─ batch-fetch holding prices
    ├─ fetch key indices
    ├─ fetch watchlist prices
    ├─ compute alerts (big movers, drawdowns)
    └─ return BriefingReport dict
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime

from agent.user_profile import get_or_create_profile

# Public helpers from batch_tools
from agent.batch_tools import fetch_latest_price, KEY_INDICES

ALERT_DAY_CHANGE_PCT = 5.0      # % daily move to flag (pct_chg is already in %)
ALERT_DRAWDOWN_RATIO = -0.10    # decimal drawdown from cost to flag (e.g. -0.10 = -10%)


@dataclass
class BriefingReport:
    """Structured daily briefing output."""
    generated_at: str = ""
    user_id: str = ""

    # Market overview
    indices: list[dict] = field(default_factory=list)

    # Holdings live
    holdings: list[dict] = field(default_factory=list)
    portfolio_value_est: float = 0.0
    portfolio_pnl_est: float = 0.0

    # Watchlist
    watchlist: list[dict] = field(default_factory=list)

    # Alerts
    alerts: list[str] = field(default_factory=list)

    # Preferences reminder
    preferences_summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def run_daily_briefing(user_id: str = "dev_user") -> dict:
    """Generate a daily briefing for the given user.

    Returns a dict (BriefingReport) that can be serialised or
    passed to the LLM for natural-language formatting.
    """
    profile = get_or_create_profile(user_id)
    report = BriefingReport(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        user_id=user_id,
    )

    # ---------------------------------------------------------------
    # 1. Key indices
    # ---------------------------------------------------------------
    for idx_code, idx_name in KEY_INDICES:
        quote = fetch_latest_price(idx_code, "index")
        if quote.get("found"):
            report.indices.append({
                "name": idx_name,
                "ts_code": idx_code,
                "close": quote.get("close"),
                "pct_chg": quote.get("pct_chg"),
                "trade_date": quote.get("trade_date"),
            })

    # ---------------------------------------------------------------
    # 2. Portfolio holdings
    # ---------------------------------------------------------------
    total_live_value = 0.0
    total_cost_value = 0.0
    alerts: list[str] = []

    for h in profile.holdings:
        if not h.ts_code:
            report.holdings.append({
                "name": h.name,
                "status": "unresolved",
                "shares": h.shares,
                "cost_price": h.cost_price,
            })
            continue

        quote = fetch_latest_price(h.ts_code, h.asset_type)
        live_price = quote.get("close", h.current_price)
        day_chg = quote.get("pct_chg", 0.0) or 0.0

        # Decimal ratio (0.15 = +15%) — consistent with batch_tools
        cost_pnl_pct = 0.0
        if h.cost_price and h.cost_price > 0 and live_price and live_price > 0:
            cost_pnl_pct = (live_price - h.cost_price) / h.cost_price

        live_value = (h.shares or 0) * (live_price or 0)
        cost_value = (h.shares or 0) * (h.cost_price or 0)
        total_live_value += live_value
        total_cost_value += cost_value

        entry = {
            "name": h.name,
            "ts_code": h.ts_code,
            "asset_type": h.asset_type,
            "shares": h.shares,
            "cost_price": h.cost_price,
            "live_price": live_price,
            "day_chg_pct": round(day_chg, 2),
            "cost_pnl_pct": round(cost_pnl_pct, 5),
            "live_value": round(live_value, 2),
            "trade_date": quote.get("trade_date"),
        }
        report.holdings.append(entry)

        # Alert: big daily move
        if abs(day_chg) >= ALERT_DAY_CHANGE_PCT:
            direction = "📈 涨" if day_chg > 0 else "📉 跌"
            alerts.append(f"{h.name} 今日{direction}{abs(day_chg):.1f}%")

        # Alert: large drawdown from cost
        if cost_pnl_pct <= ALERT_DRAWDOWN_RATIO:
            alerts.append(f"{h.name} 持仓浮亏 {cost_pnl_pct * 100:.1f}%（成本 {h.cost_price}）")

    report.portfolio_value_est = round(total_live_value, 2)
    report.portfolio_pnl_est = round(total_live_value - total_cost_value, 2)

    # ---------------------------------------------------------------
    # 3. Watchlist
    # ---------------------------------------------------------------
    for w in profile.watchlist:
        if not w.ts_code:
            report.watchlist.append({"name": w.name, "status": "unresolved"})
            continue

        quote = fetch_latest_price(w.ts_code, "stock")
        report.watchlist.append({
            "name": w.name,
            "ts_code": w.ts_code,
            "close": quote.get("close"),
            "pct_chg": quote.get("pct_chg"),
            "trade_date": quote.get("trade_date"),
            "reason": w.reason,
        })

        day_chg = quote.get("pct_chg", 0.0) or 0.0
        if abs(day_chg) >= ALERT_DAY_CHANGE_PCT:
            direction = "📈" if day_chg > 0 else "📉"
            alerts.append(f"[关注] {w.name} {direction}{abs(day_chg):.1f}%")

    report.alerts = alerts

    # ---------------------------------------------------------------
    # 4. Preferences reminder
    # ---------------------------------------------------------------
    prefs = profile.preferences
    parts = []
    if prefs.risk_tolerance:
        parts.append(f"风险偏好: {prefs.risk_tolerance}")
    if prefs.investment_horizon:
        parts.append(f"投资周期: {prefs.investment_horizon}")
    if prefs.max_single_position_pct:
        parts.append(f"单只上限: {prefs.max_single_position_pct}%")
    report.preferences_summary = " | ".join(parts) if parts else ""

    return report.to_dict()
