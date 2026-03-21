"""Watchlist alert collector — price changes and threshold alerts."""

from __future__ import annotations

import logging

from spiderman.core.config import SpidermanConfig
from spiderman.core.events import Event

from agent.notifications.collectors.base import QuantBaseCollector

logger = logging.getLogger(__name__)

# Default alert thresholds
_DEFAULT_PCT_THRESHOLD = 3.0  # Alert if daily change > 3%


class WatchlistAlertCollector(QuantBaseCollector):
    default_cron = "0 15 * * 1-5"

    def __init__(self, config: SpidermanConfig) -> None:
        super().__init__(config)
        self._store = None

    @property
    def name(self) -> str:
        return "watchlist_alert"

    def _get_store(self):
        if self._store is None:
            self._store = self._open_store()
        return self._store

    def collect(self, user_id: str | None = None) -> list[Event]:
        user_ids = [user_id] if user_id else self._list_user_ids()
        events: list[Event] = []

        for uid in user_ids:
            profile = self._load_user_profile(uid)
            if not profile:
                continue

            watchlist = profile.get("watchlist", [])
            if not watchlist:
                continue

            for item in watchlist:
                try:
                    event = self._check_item(uid, item)
                    if event:
                        events.append(event)
                except Exception:
                    logger.debug(
                        "Watchlist check failed for %s/%s",
                        uid,
                        item.get("ts_code", "?"),
                    )

        return events

    def _check_item(self, user_id: str, item: dict) -> Event | None:
        ts_code = item.get("ts_code", "")
        if not ts_code:
            return None

        store = self._get_store()

        # Try stock daily first, then ETF
        df = None
        for method_name in ("daily", "etf_daily", "index_daily"):
            try:
                method = getattr(store, method_name, None)
                if method is None:
                    continue
                df = method(ts_code)
                if df is not None and not df.empty:
                    break
            except Exception:
                continue

        if df is None or df.empty:
            return None

        df = df.sort_values("trade_date")
        latest = df.iloc[-1]
        pct_chg = float(latest.get("pct_chg", 0))
        close = float(latest.get("close", 0))
        trade_date = str(latest.get("trade_date", ""))

        alerts: list[str] = []

        # Check daily percentage change threshold
        if abs(pct_chg) >= _DEFAULT_PCT_THRESHOLD:
            direction = "涨" if pct_chg > 0 else "跌"
            alerts.append(f"日{direction}幅 {abs(pct_chg):.2f}%")

        # Check for new high/low (20-day window)
        if len(df) >= 20:
            window = df.tail(20)
            high_20 = window["close"].max()
            low_20 = window["close"].min()
            if close >= high_20:
                alerts.append("20日新高")
            elif close <= low_20:
                alerts.append("20日新低")

        # Check volume spike (2x average of last 20 days)
        if len(df) >= 21 and "vol" in df.columns:
            avg_vol = df.tail(21).head(20)["vol"].mean()
            latest_vol = float(latest.get("vol", 0))
            if avg_vol > 0 and latest_vol > 2 * avg_vol:
                alerts.append(f"成交量异动 ({latest_vol / avg_vol:.1f}x)")

        if not alerts:
            return None

        return Event(
            event_type="watchlist_alert",
            source=self.name,
            user_id=user_id,
            priority=1,
            payload={
                "name": item.get("name", ts_code),
                "ts_code": ts_code,
                "reason": item.get("reason", ""),
                "trade_date": trade_date,
                "close": close,
                "pct_chg": pct_chg,
                "alerts": alerts,
            },
        )
