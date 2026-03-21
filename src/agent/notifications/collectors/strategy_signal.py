"""Strategy signal collector — detect entry/exit signals from user strategies."""

from __future__ import annotations

import logging

import pandas as pd

from spiderman.core.config import SpidermanConfig
from spiderman.core.events import Event

from agent.notifications.collectors.base import QuantBaseCollector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------


def _signals_dual_ma(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    fast = int(params.get("fast", 5))
    slow = int(params.get("slow", 20))
    ma_type = str(params.get("ma_type", "sma")).lower()
    close = df["close"]
    if ma_type == "ema":
        ma_fast = close.ewm(span=fast, adjust=False).mean()
        ma_slow = close.ewm(span=slow, adjust=False).mean()
    else:
        ma_fast = close.rolling(fast).mean()
        ma_slow = close.rolling(slow).mean()
    entry = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
    exit_ = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))
    return entry, exit_


def _signals_bollinger(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    period = int(params.get("period", 20))
    num_std = float(params.get("num_std", 2.0))
    close = df["close"]
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    lower = ma - num_std * std
    upper = ma + num_std * std
    entry = (close <= lower) & (close.shift(1) > lower.shift(1))
    exit_ = (close >= upper) & (close.shift(1) < upper.shift(1))
    return entry, exit_


def _signals_macd(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    fast = int(params.get("fast", 12))
    slow = int(params.get("slow", 26))
    signal = int(params.get("signal", 9))
    close = df["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    entry = (hist > 0) & (hist.shift(1) <= 0)
    exit_ = (hist < 0) & (hist.shift(1) >= 0)
    return entry, exit_


_SIGNAL_REGISTRY = {
    "dual_ma": _signals_dual_ma,
    "bollinger": _signals_bollinger,
    "macd": _signals_macd,
}

_DEFAULT_PARAMS = {
    "dual_ma": {"fast": 5, "slow": 20, "ma_type": "sma"},
    "bollinger": {"period": 20, "num_std": 2.0},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
}


class StrategySignalCollector(QuantBaseCollector):
    default_cron = "0 17 * * 1-5"

    def __init__(self, config: SpidermanConfig) -> None:
        super().__init__(config)
        self._store = None

    @property
    def name(self) -> str:
        return "strategy_signal"

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

            strategies = profile.get("strategies", [])
            if not strategies:
                continue

            for strat in strategies:
                try:
                    new_events = self._check_strategy(uid, strat)
                    events.extend(new_events)
                except Exception:
                    logger.exception(
                        "Failed strategy check %s for user %s",
                        strat.get("name", "?"),
                        uid,
                    )

        return events

    def _check_strategy(self, user_id: str, strategy: dict) -> list[Event]:
        """Check if a strategy fires a signal today for any of its ts_codes."""
        strat_name = strategy.get("name", "")
        if strat_name not in _SIGNAL_REGISTRY:
            return []

        ts_codes = strategy.get("ts_codes", [])
        if not ts_codes:
            return []

        merged_params = dict(_DEFAULT_PARAMS.get(strat_name, {}))
        merged_params.update(strategy.get("params", {}))

        signal_fn = _SIGNAL_REGISTRY[strat_name]
        store = self._get_store()
        events: list[Event] = []

        for ts_code in ts_codes:
            try:
                df = store.daily_adj(ts_code, how="hfq")
                if df is None or len(df) < 30:
                    continue
                df = df.sort_values("trade_date").reset_index(drop=True)

                entry_sig, exit_sig = signal_fn(df, merged_params)

                # Check latest bar for signal
                last_entry = bool(entry_sig.iloc[-1]) if not entry_sig.empty else False
                last_exit = bool(exit_sig.iloc[-1]) if not exit_sig.empty else False

                if not last_entry and not last_exit:
                    continue

                signal_type = "ENTRY" if last_entry else "EXIT"
                last_row = df.iloc[-1]

                events.append(Event(
                    event_type="strategy_signal",
                    source=self.name,
                    user_id=user_id,
                    priority=2 if last_entry else 1,
                    payload={
                        "signal_type": signal_type,
                        "strategy": strat_name,
                        "params": merged_params,
                        "ts_code": ts_code,
                        "trade_date": str(last_row.get("trade_date", "")),
                        "close": float(last_row.get("close", 0)),
                        "pct_chg": float(last_row.get("pct_chg", 0)),
                    },
                ))

            except Exception:
                logger.debug("Signal check failed for %s/%s", ts_code, strat_name)

        return events
