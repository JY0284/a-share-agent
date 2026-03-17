"""Tests for the UserProfile structured storage layer.

Covers: CRUD, persistence round-trip, diff tracking, formatting.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Patch storage dir via environment variable BEFORE import
_tmp_dir = tempfile.mkdtemp()
_profiles_dir = os.path.join(_tmp_dir, "user_profiles")
os.environ["USER_PROFILE_DIR"] = _profiles_dir

import agent.user_profile as up


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_profile_dir():
    """Clean the temp profile dir before each test."""
    # Ensure the module picks up our temp dir (may have been cached by another test)
    os.environ["USER_PROFILE_DIR"] = _profiles_dir
    up._profile_dir_ensured = None  # reset cache
    d = Path(_profiles_dir)
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.json"):
        f.unlink()
    yield


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestHolding:
    def test_defaults(self):
        h = up.Holding(name="贵州茅台")
        assert h.ts_code == ""
        assert h.asset_type == "stock"
        assert h.shares == 0
        assert h.pnl_pct == 0.0

    def test_full(self):
        h = up.Holding(
            name="贵州茅台",
            ts_code="600519.SH",
            shares=100,
            cost_price=1800.0,
            current_price=1900.0,
            market_value=190000.0,
            pnl_pct=0.0556,
            tags=["白酒", "消费"],
        )
        assert h.ts_code == "600519.SH"
        assert h.tags == ["白酒", "消费"]


class TestUserPreferences:
    def test_defaults(self):
        p = up.UserPreferences()
        assert p.risk_tolerance == "moderate"
        assert p.max_single_position_pct == 0.25

    def test_custom(self):
        p = up.UserPreferences(risk_tolerance="conservative", investment_horizon="long")
        assert p.risk_tolerance == "conservative"


class TestUserProfile:
    def test_empty_profile(self):
        profile = up.UserProfile(user_id="test_user_1")
        assert profile.user_id == "test_user_1"
        assert profile.holdings == []
        assert profile.total_assets == 0.0

    def test_with_holdings(self):
        h = up.Holding(name="茅台", ts_code="600519.SH", shares=100)
        profile = up.UserProfile(user_id="u1", holdings=[h], cash=50000, total_assets=250000)
        assert len(profile.holdings) == 1
        assert profile.cash == 50000


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load(self):
        profile = up.UserProfile(user_id="persist_1")
        profile.holdings.append(up.Holding(name="茅台", ts_code="600519.SH"))
        profile.cash = 10000.0

        up.save_profile(profile)

        loaded = up.load_profile("persist_1")
        assert loaded is not None
        assert loaded.user_id == "persist_1"
        assert loaded.cash == 10000.0
        assert len(loaded.holdings) == 1
        assert loaded.holdings[0].ts_code == "600519.SH"

    def test_load_nonexistent(self):
        result = up.load_profile("no_such_user")
        assert result is None

    def test_get_or_create(self):
        profile = up.get_or_create_profile("new_user_42")
        assert profile.user_id == "new_user_42"
        # Should have been saved
        assert (Path(_profiles_dir) / "new_user_42.json").exists()

    def test_json_round_trip(self):
        profile = up.UserProfile(
            user_id="rt_user",
            holdings=[
                up.Holding(name="A", ts_code="000001.SZ", shares=500, cost_price=10.0),
                up.Holding(name="B", ts_code="600000.SH", shares=200),
            ],
            watchlist=[up.WatchlistItem(name="C", ts_code="300750.SZ", reason="电池龙头")],
            preferences=up.UserPreferences(risk_tolerance="aggressive"),
            cash=88888.0,
            total_assets=200000.0,
        )
        up.save_profile(profile)

        raw = json.loads((Path(_profiles_dir) / "rt_user.json").read_text(encoding="utf-8"))
        assert raw["user_id"] == "rt_user"
        assert len(raw["holdings"]) == 2
        assert raw["preferences"]["risk_tolerance"] == "aggressive"


# ---------------------------------------------------------------------------
# CRUD tests
# ---------------------------------------------------------------------------


class TestCRUD:
    def test_update_portfolio(self):
        up.get_or_create_profile("crud_1")
        holdings = [
            {"name": "茅台", "ts_code": "600519.SH", "shares": 100, "cost_price": 1800},
            {"name": "宁德时代", "ts_code": "300750.SZ", "shares": 50, "cost_price": 200},
        ]
        result = up.update_portfolio("crud_1", holdings=holdings, cash=50000, total_assets=300000)
        assert result["saved"] is True

        # Should be persisted
        reloaded = up.load_profile("crud_1")
        assert len(reloaded.holdings) == 2
        assert reloaded.cash == 50000

    def test_update_portfolio_creates_snapshot(self):
        up.get_or_create_profile("snap_1")
        h1 = [{"name": "A", "ts_code": "000001.SZ", "shares": 100}]
        up.update_portfolio("snap_1", holdings=h1)

        h2 = [{"name": "B", "ts_code": "000002.SZ", "shares": 200}]
        up.update_portfolio("snap_1", holdings=h2)

        # Should have 1 snapshot (from the first → second update)
        reloaded = up.load_profile("snap_1")
        assert len(reloaded.snapshots) >= 1

    def test_update_preferences(self):
        up.get_or_create_profile("pref_1")
        up.update_preferences("pref_1", risk_tolerance="conservative", investment_horizon="long")
        reloaded = up.load_profile("pref_1")
        assert reloaded.preferences.risk_tolerance == "conservative"
        assert reloaded.preferences.investment_horizon == "long"

    def test_add_remove_watchlist(self):
        up.get_or_create_profile("wl_1")
        up.add_watchlist_item("wl_1", {"name": "宁德时代", "ts_code": "300750.SZ", "reason": "电池"})
        profile = up.load_profile("wl_1")
        assert len(profile.watchlist) == 1

        up.add_watchlist_item("wl_1", {"name": "比亚迪", "ts_code": "002594.SZ", "reason": "新能源车"})
        profile = up.load_profile("wl_1")
        assert len(profile.watchlist) == 2

        up.remove_watchlist_item("wl_1", "300750.SZ")
        profile = up.load_profile("wl_1")
        assert len(profile.watchlist) == 1
        assert profile.watchlist[0].ts_code == "002594.SZ"

    def test_add_strategy(self):
        up.get_or_create_profile("strat_1")
        up.add_strategy("strat_1", {
            "name": "双均线",
            "description": "MA5/MA20 crossover",
            "params": {"fast": 5, "slow": 20},
        })
        profile = up.load_profile("strat_1")
        assert len(profile.strategies) == 1
        assert profile.strategies[0].params == {"fast": 5, "slow": 20}


# ---------------------------------------------------------------------------
# Formatting tests
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_format_empty_profile(self):
        profile = up.UserProfile(user_id="fmt_empty")
        text = up.format_full_profile_context(profile)
        assert "No portfolio data" in text or text == ""

    def test_format_with_holdings(self):
        profile = up.UserProfile(
            user_id="fmt_full",
            holdings=[
                up.Holding(name="茅台", ts_code="600519.SH", shares=100, cost_price=1800, market_value=190000),
            ],
            cash=50000,
            total_assets=240000,
        )
        text = up.format_full_profile_context(profile)
        assert "茅台" in text
        assert "600519" in text

    def test_format_portfolio_summary(self):
        profile = up.UserProfile(
            user_id="fmt_ps",
            holdings=[up.Holding(name="X", ts_code="000001.SZ", shares=100)],
            total_assets=100000,
        )
        summary = up.format_portfolio_summary(profile)
        assert "000001" in summary
