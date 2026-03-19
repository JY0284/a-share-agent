"""Tests for batch_tools module.

These tests mock the stock_data layer so they run without a real data store.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Isolate profile storage
_tmp = tempfile.mkdtemp()
os.environ["USER_PROFILE_DIR"] = os.path.join(_tmp, "user_profiles")

import agent.user_profile as up
import agent.batch_tools as bt


# ---------------------------------------------------------------------------
# Fixtures & mocks
# ---------------------------------------------------------------------------

_FAKE_PRICE_ROW = {
    "trade_date": "20250314",
    "open": 100.0,
    "high": 102.0,
    "low": 99.0,
    "close": 101.5,
    "pct_chg": 1.5,
    "vol": 10000,
}


def _mock_get_daily_prices(ts_code, **kwargs):
    return {"rows": [{"ts_code": ts_code, **_FAKE_PRICE_ROW}]}


def _mock_get_index_daily_prices(ts_code, **kwargs):
    return {"rows": [{"ts_code": ts_code, **_FAKE_PRICE_ROW, "close": 3200.0, "pct_chg": 0.8}]}


def _mock_get_etf_daily_prices(ts_code, **kwargs):
    return {"rows": [{"ts_code": ts_code, **_FAKE_PRICE_ROW, "close": 1.25, "pct_chg": -0.3}]}


@pytest.fixture(autouse=True)
def _mock_data_layer():
    """Patch all stock_data.agent_tools calls."""
    with patch.object(bt, "get_daily_prices", side_effect=_mock_get_daily_prices), \
         patch.object(bt, "get_index_daily_prices", side_effect=_mock_get_index_daily_prices), \
         patch.object(bt, "get_etf_daily_prices", side_effect=_mock_get_etf_daily_prices):
        yield


@pytest.fixture(autouse=True)
def _clean_profiles():
    up._profile_dir_ensured = None  # reset cache
    d = Path(os.environ["USER_PROFILE_DIR"])
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.json"):
        f.unlink()
    yield


# ---------------------------------------------------------------------------
# tool_batch_quotes
# ---------------------------------------------------------------------------


class TestBatchQuotes:
    def test_basic(self):
        result = bt.tool_batch_quotes.invoke({
            "ts_codes": ["600519.SH", "000001.SZ"],
        })
        assert result["count"] == 2
        assert result["quotes"][0]["found"] is True
        assert result["quotes"][0]["close"] == 101.5

    def test_with_explicit_types(self):
        result = bt.tool_batch_quotes.invoke({
            "ts_codes": ["000300.SH", "510300.SH"],
        })
        assert result["count"] == 2
        # Index uses mock that returns 3200
        assert result["quotes"][0]["close"] == 3200.0
        # ETF uses mock that returns 1.25
        assert result["quotes"][1]["close"] == 1.25

    def test_empty_list(self):
        result = bt.tool_batch_quotes.invoke({"ts_codes": []})
        assert result["count"] == 0
        assert result["quotes"] == []


# ---------------------------------------------------------------------------
# tool_portfolio_live_snapshot
# ---------------------------------------------------------------------------


class TestPortfolioLiveSnapshot:
    def test_no_portfolio(self):
        """When user has no saved holdings, should indicate that."""
        result = bt.tool_portfolio_live_snapshot.invoke(
            {},
            config={"configurable": {"user_id": "empty_user"}},
        )
        assert result["has_portfolio"] is False

    def test_with_holdings(self):
        """When user has holdings, should fetch live prices for all."""
        up.get_or_create_profile("test_snap_user")
        up.update_portfolio(
            "test_snap_user",
            holdings=[
                {"name": "茅台", "ts_code": "600519.SH", "asset_type": "stock", "shares": 100, "cost_price": 1800},
                {"name": "沪深300ETF", "ts_code": "510300.SH", "asset_type": "etf", "shares": 1000, "cost_price": 4.0},
            ],
            cash=50000,
            total_assets=300000,
        )

        result = bt.tool_portfolio_live_snapshot.invoke(
            {},
            config={"configurable": {"user_id": "test_snap_user"}},
        )
        assert result["has_portfolio"] is True
        assert len(result["holdings_live"]) == 2
        assert result["holdings_live"][0]["live_price"] == 101.5  # stock mock
        assert result["holdings_live"][1]["live_price"] == 1.25   # etf mock
        assert len(result["indices"]) == 5  # 5 key indices
        assert "portfolio_summary" in result


# ---------------------------------------------------------------------------
# tool_market_overview
# ---------------------------------------------------------------------------


class TestMarketOverview:
    def test_returns_indices(self):
        with patch.object(bt, "get_fx_daily", return_value={"rows": [{"close": 7.23}]}), \
             patch.object(bt, "get_lpr", return_value={"rows": [{"rate": 3.45}]}), \
             patch.object(bt, "get_cpi", return_value={"rows": [{"nt_val": 102.1}]}), \
             patch.object(bt, "get_cn_m", return_value={"rows": [{"m2_yoy": 8.7}]}):
            result = bt.tool_market_overview.invoke({})

        assert len(result["indices"]) == 5
        assert result["indices"][0]["name"] == "上证指数"
        assert "fx" in result
        assert "macro" in result
