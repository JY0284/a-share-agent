"""Portfolio report collector — daily P&L summary for each user."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from spiderman.core.config import SpidermanConfig
from spiderman.core.events import Event

from agent.notifications.collectors.base import QuantBaseCollector

logger = logging.getLogger(__name__)

# Key market indices to include alongside portfolio
KEY_INDICES = [
    ("000001.SH", "上证指数"),
    ("399001.SZ", "深证成指"),
    ("399006.SZ", "创业板指"),
    ("000300.SH", "沪深300"),
    ("000905.SH", "中证500"),
]

# Mapping from asset_type → store method hint
_ASSET_TYPE_FETCHERS = {
    "stock": "daily",
    "etf": "etf_daily",
    "index_etf": "etf_daily",
    "qdii": "etf_daily",
    "bond_fund": "etf_daily",
    "index": "index_daily",
}


class PortfolioReportCollector(QuantBaseCollector):
    default_cron = "0 18 * * 1-5"

    def __init__(self, config: SpidermanConfig) -> None:
        super().__init__(config)
        self._store = None

    @property
    def name(self) -> str:
        return "portfolio_report"

    def _get_store(self):
        if self._store is None:
            self._store = self._open_store()
        return self._store

    def _fetch_latest_price(self, ts_code: str, asset_type: str = "stock") -> dict:
        """Fetch the latest price row for a given ts_code."""
        store = self._get_store()
        fetcher_order = ["daily", "etf_daily", "index_daily"]
        hint = _ASSET_TYPE_FETCHERS.get(asset_type, "daily")
        fetcher_order.sort(key=lambda x: (0 if x == hint else 1))

        for method_name in fetcher_order:
            try:
                method = getattr(store, method_name, None)
                if method is None:
                    continue
                df = method(ts_code)
                if df is not None and not df.empty:
                    df = df.sort_values("trade_date")
                    row = df.iloc[-1]
                    return {
                        "found": True,
                        "close": float(row.get("close", 0)),
                        "pct_chg": float(row.get("pct_chg", 0)),
                        "trade_date": str(row.get("trade_date", "")),
                    }
            except Exception:
                continue
        return {"found": False}

    def collect(self, user_id: str | None = None) -> list[Event]:
        user_ids = [user_id] if user_id else self._list_user_ids()
        events: list[Event] = []

        for uid in user_ids:
            profile = self._load_user_profile(uid)
            if not profile or not profile.get("holdings"):
                continue

            try:
                event = self._build_report(uid, profile)
                if event:
                    events.append(event)
            except Exception:
                logger.exception("Failed to build portfolio report for %s", uid)

        return events

    def _build_report(self, user_id: str, profile: dict) -> Event | None:
        holdings = profile.get("holdings", [])
        if not holdings:
            return None

        # Parallel-fetch prices for all holdings + indices
        holdings_live = []

        with ThreadPoolExecutor(max_workers=min(len(holdings) + len(KEY_INDICES), 10)) as pool:
            # Submit holding fetches
            h_futures = {}
            for h in holdings:
                ts_code = h.get("ts_code", "")
                if ts_code:
                    future = pool.submit(
                        self._fetch_latest_price, ts_code, h.get("asset_type", "stock")
                    )
                    h_futures[future] = h

            # Submit index fetches
            idx_futures = {}
            for idx_code, idx_name in KEY_INDICES:
                future = pool.submit(self._fetch_latest_price, idx_code, "index")
                idx_futures[future] = (idx_code, idx_name)

            # Collect holding results
            for future in as_completed(h_futures):
                h = h_futures[future]
                quote = future.result()
                live_price = quote.get("close", h.get("current_price", 0))
                cost_price = h.get("cost_price", 0)
                shares = h.get("shares", 0)

                if cost_price > 0 and live_price > 0:
                    cost_pnl_pct = (live_price - cost_price) / cost_price
                else:
                    cost_pnl_pct = h.get("pnl_pct", 0)

                live_value = shares * live_price if shares and live_price else h.get("market_value", 0)

                holdings_live.append({
                    "name": h.get("name", ""),
                    "ts_code": h.get("ts_code", ""),
                    "asset_type": h.get("asset_type", "stock"),
                    "shares": shares,
                    "cost_price": cost_price,
                    "live_price": live_price,
                    "live_date": quote.get("trade_date", ""),
                    "day_chg_pct": quote.get("pct_chg", 0),
                    "cost_pnl_pct": round(cost_pnl_pct, 5),
                    "live_value": round(live_value, 2),
                    "tags": h.get("tags", []),
                })

            # Collect index results
            indices = []
            for future in as_completed(idx_futures):
                idx_code, idx_name = idx_futures[future]
                quote = future.result()
                if quote.get("found"):
                    indices.append({
                        "name": idx_name,
                        "ts_code": idx_code,
                        "close": quote.get("close"),
                        "pct_chg": quote.get("pct_chg"),
                        "trade_date": quote.get("trade_date"),
                    })

        # Summary
        total_live_value = sum(h.get("live_value", 0) for h in holdings_live)
        total_cost = sum(
            h.get("shares", 0) * h.get("cost_price", 0)
            for h in holdings_live
            if h.get("cost_price", 0) > 0
        )
        total_pnl = total_live_value - total_cost if total_cost > 0 else 0

        return Event(
            event_type="portfolio_report",
            source=self.name,
            user_id=user_id,
            payload={
                "holdings_live": holdings_live,
                "indices": indices,
                "portfolio_summary": {
                    "total_assets": profile.get("total_assets", 0),
                    "cash": profile.get("cash", 0),
                    "live_market_value": round(total_live_value, 2),
                    "total_pnl": round(total_pnl, 2),
                    "holding_count": len(holdings_live),
                },
                "preferences": profile.get("preferences", {}),
            },
        )
