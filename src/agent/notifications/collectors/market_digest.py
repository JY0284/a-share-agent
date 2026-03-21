"""Market digest collector — broad market overview for all subscribers."""

from __future__ import annotations

import logging

from spiderman.core.config import SpidermanConfig
from spiderman.core.events import Event

from agent.notifications.collectors.base import QuantBaseCollector

logger = logging.getLogger(__name__)

KEY_INDICES = [
    ("000001.SH", "上证指数"),
    ("399001.SZ", "深证成指"),
    ("399006.SZ", "创业板指"),
    ("000300.SH", "沪深300"),
    ("000905.SH", "中证500"),
]


class MarketDigestCollector(QuantBaseCollector):
    default_cron = "30 8 * * 1-5"

    def __init__(self, config: SpidermanConfig) -> None:
        super().__init__(config)
        self._store = None

    @property
    def name(self) -> str:
        return "market_digest"

    def _get_store(self):
        if self._store is None:
            self._store = self._open_store()
        return self._store

    def collect(self, user_id: str | None = None) -> list[Event]:
        """Collect market overview — returns a single broadcast event."""
        try:
            payload = self._build_digest()
        except Exception:
            logger.exception("Failed to build market digest")
            return []

        return [
            Event(
                event_type="market_digest",
                source=self.name,
                user_id=user_id,  # None = broadcast to all subscribers
                payload=payload,
            )
        ]

    def _build_digest(self) -> dict:
        store = self._get_store()
        result: dict = {"indices": [], "fx": {}, "macro": {}}

        # Indices
        for idx_code, idx_name in KEY_INDICES:
            try:
                df = store.index_daily(idx_code)
                if df is not None and not df.empty:
                    df = df.sort_values("trade_date")
                    row = df.iloc[-1]
                    result["indices"].append({
                        "name": idx_name,
                        "ts_code": idx_code,
                        "close": float(row.get("close", 0)),
                        "pct_chg": float(row.get("pct_chg", 0)),
                        "trade_date": str(row.get("trade_date", "")),
                    })
            except Exception:
                logger.debug("Failed to fetch index %s", idx_code)

        # FX — USDCNH
        try:
            fx_df = store.read("fx_daily")
            if fx_df is not None and not fx_df.empty:
                usd = fx_df[fx_df["ts_code"] == "USDCNH"] if "ts_code" in fx_df.columns else fx_df
                if not usd.empty:
                    usd = usd.sort_values("trade_date")
                    row = usd.iloc[-1]
                    result["fx"] = {
                        "pair": "USDCNH",
                        "close": float(row.get("close", 0)),
                        "pct_chg": float(row.get("pct_chg", 0)),
                        "trade_date": str(row.get("trade_date", "")),
                    }
        except Exception:
            logger.debug("Failed to fetch FX data")

        # Macro — LPR
        try:
            lpr = store.read("lpr")
            if lpr is not None and not lpr.empty:
                lpr = lpr.sort_values("date") if "date" in lpr.columns else lpr
                row = lpr.iloc[-1]
                result["macro"]["lpr_latest"] = {
                    "date": str(row.get("date", "")),
                    "lpr_1y": float(row.get("lpr_1y", 0)) if "lpr_1y" in row.index else None,
                    "lpr_5y": float(row.get("lpr_5y", 0)) if "lpr_5y" in row.index else None,
                }
        except Exception:
            logger.debug("Failed to fetch LPR data")

        # Macro — CPI
        try:
            cpi = store.read("cpi")
            if cpi is not None and not cpi.empty:
                cpi = cpi.sort_values("month") if "month" in cpi.columns else cpi
                row = cpi.iloc[-1]
                result["macro"]["cpi_latest"] = {
                    "month": str(row.get("month", "")),
                    "nt_yoy": float(row.get("nt_yoy", 0)) if "nt_yoy" in row.index else None,
                }
        except Exception:
            logger.debug("Failed to fetch CPI data")

        # Macro — Money supply (M0/M1/M2)
        try:
            cn_m = store.read("cn_m")
            if cn_m is not None and not cn_m.empty:
                cn_m = cn_m.sort_values("month") if "month" in cn_m.columns else cn_m
                row = cn_m.iloc[-1]
                result["macro"]["money_supply_latest"] = {
                    "month": str(row.get("month", "")),
                    "m0": float(row.get("m0", 0)) if "m0" in row.index else None,
                    "m1": float(row.get("m1", 0)) if "m1" in row.index else None,
                    "m2": float(row.get("m2", 0)) if "m2" in row.index else None,
                    "m1_yoy": float(row.get("m1_yoy", 0)) if "m1_yoy" in row.index else None,
                    "m2_yoy": float(row.get("m2_yoy", 0)) if "m2_yoy" in row.index else None,
                }
        except Exception:
            logger.debug("Failed to fetch money supply data")

        return result
