"""LangChain tool wrappers for the stock_data agent_tools."""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.tools import tool

from stock_data.agent_tools import (
    get_adj_factor,
    get_daily_adj_prices,
    get_daily_basic,
    get_daily_prices,
    get_monthly_prices,
    get_namechange,
    get_new_share,
    get_next_trade_date,
    get_prev_trade_date,
    get_stock_basic,
    get_stock_company,
    get_stk_limit,
    get_suspend_d,
    get_trade_cal,
    get_trading_days,
    get_universe,
    get_weekly_prices,
    is_trading_day,
    query_dataset,
    resolve_symbol,
)

# Get store directory from environment or use default
STORE_DIR = os.environ.get("STOCK_DATA_STORE_DIR", "../stock_data/store")


# -----------------------------------------------------------------------------
# Identity / Universe Tools
# -----------------------------------------------------------------------------


@tool
def tool_resolve_symbol(symbol_or_ts_code: str) -> dict:
    """Resolve a stock symbol (e.g. '300888') or ts_code (e.g. '300888.SZ') to full information.
    
    Use this to convert user-provided stock codes to the standard ts_code format.
    Returns: symbol, ts_code, and list_date.
    
    Args:
        symbol_or_ts_code: The stock symbol or ts_code to resolve
    """
    return resolve_symbol(symbol_or_ts_code, store_dir=STORE_DIR)


@tool
def tool_get_stock_basic(
    ts_code: str | None = None,
    symbol: str | None = None,
    limit: int | None = None,
) -> dict:
    """Get stock basic information including name, industry, exchange, list status, etc.
    
    Can filter by ts_code or symbol. Without filters, returns all stocks.
    
    Args:
        ts_code: Filter by ts_code (e.g., '000001.SZ')
        symbol: Filter by symbol (e.g., '000001')
        limit: Maximum number of results to return
    """
    return get_stock_basic(ts_code=ts_code, symbol=symbol, limit=limit, store_dir=STORE_DIR)


@tool
def tool_get_stock_company(ts_code: str) -> dict:
    """Get detailed company profile including chairman, employees, main business, etc.
    
    Args:
        ts_code: The stock ts_code (e.g., '000001.SZ')
    """
    return get_stock_company(ts_code, store_dir=STORE_DIR)


@tool
def tool_get_universe(
    list_status: str | None = "L",
    exchange: str | None = None,
    market: str | None = None,
    industry: str | None = None,
    area: str | None = None,
    limit: int | None = None,
) -> dict:
    """Get filtered universe of stocks.
    
    Args:
        list_status: 'L' for listed, 'D' for delisted, 'P' for paused. Default 'L'.
        exchange: Filter by exchange ('SSE' for Shanghai, 'SZSE' for Shenzhen)
        market: Filter by market (e.g., '主板', '创业板', '科创板')
        industry: Filter by industry name
        area: Filter by region/province
        limit: Maximum number of results to return
    """
    return get_universe(
        list_status=list_status,
        exchange=exchange,
        market=market,
        industry=industry,
        area=area,
        limit=limit,
        store_dir=STORE_DIR,
    )


# -----------------------------------------------------------------------------
# Calendar Tools
# -----------------------------------------------------------------------------


@tool
def tool_get_trading_days(start_date: str, end_date: str) -> dict:
    """Get list of trading days in a date range.
    
    Args:
        start_date: Start date in YYYYMMDD format (e.g., '20240101')
        end_date: End date in YYYYMMDD format (e.g., '20240131')
    """
    return get_trading_days(start_date, end_date, store_dir=STORE_DIR)


@tool
def tool_is_trading_day(date: str) -> dict:
    """Check if a specific date is a trading day.
    
    Args:
        date: Date to check in YYYYMMDD format (e.g., '20240115')
    """
    return is_trading_day(date, store_dir=STORE_DIR)


@tool
def tool_get_prev_trade_date(date: str) -> dict:
    """Get the previous trading day before a given date.
    
    Args:
        date: Reference date in YYYYMMDD format
    """
    return get_prev_trade_date(date, store_dir=STORE_DIR)


@tool
def tool_get_next_trade_date(date: str) -> dict:
    """Get the next trading day after a given date.
    
    Args:
        date: Reference date in YYYYMMDD format
    """
    return get_next_trade_date(date, store_dir=STORE_DIR)


# -----------------------------------------------------------------------------
# IPO / Events Tools
# -----------------------------------------------------------------------------


@tool
def tool_get_new_share(
    year: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    ts_code: str | None = None,
    limit: int | None = None,
) -> dict:
    """Get IPO / new share information.
    
    Args:
        year: Filter by IPO year (e.g., 2024)
        start_date: Filter IPOs from this date (YYYYMMDD)
        end_date: Filter IPOs until this date (YYYYMMDD)
        ts_code: Filter by specific ts_code
        limit: Maximum number of results
    """
    return get_new_share(
        year=year,
        start_date=start_date,
        end_date=end_date,
        ts_code=ts_code,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_namechange(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Get stock name change history.
    
    Args:
        ts_code: The stock ts_code (e.g., '000001.SZ')
        start_date: Filter changes from this date
        end_date: Filter changes until this date
    """
    return get_namechange(ts_code, start_date=start_date, end_date=end_date, store_dir=STORE_DIR)


# -----------------------------------------------------------------------------
# Market Data Tools
# -----------------------------------------------------------------------------


@tool
def tool_get_daily_prices(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
) -> dict:
    """Get daily OHLCV prices (unadjusted) for a stock.
    
    Returns: trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
    
    Args:
        ts_code: The stock ts_code (e.g., '000001.SZ')
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        limit: Maximum number of results (most recent first after sorting)
    """
    return get_daily_prices(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_daily_adj_prices(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    how: Literal["qfq", "hfq", "both"] = "qfq",
    limit: int | None = None,
) -> dict:
    """Get adjusted daily prices for a stock.
    
    Adjusted prices account for stock splits and dividends.
    - qfq (前复权): Forward-adjusted, prices adjusted to latest point
    - hfq (后复权): Backward-adjusted, prices adjusted from IPO
    
    Args:
        ts_code: The stock ts_code (e.g., '000001.SZ')
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        how: Adjustment method - 'qfq' (forward), 'hfq' (backward), or 'both'
        limit: Maximum number of results
    """
    return get_daily_adj_prices(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        how=how,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_daily_basic(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
) -> dict:
    """Get daily valuation and trading metrics for a stock.
    
    Returns: PE, PE_TTM, PB, PS, dividend yield, turnover rate, volume ratio,
             total/float/free shares, total/circulating market cap
    
    Args:
        ts_code: The stock ts_code (e.g., '000001.SZ')
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        limit: Maximum number of results
    """
    return get_daily_basic(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_weekly_prices(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
) -> dict:
    """Get weekly OHLCV prices for a stock.
    
    Each row represents one trading week (ending on Friday or last trading day).
    
    Args:
        ts_code: The stock ts_code (e.g., '000001.SZ')
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        limit: Maximum number of results
    """
    return get_weekly_prices(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_monthly_prices(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
) -> dict:
    """Get monthly OHLCV prices for a stock.
    
    Each row represents one trading month (ending on last trading day of month).
    
    Args:
        ts_code: The stock ts_code (e.g., '000001.SZ')
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        limit: Maximum number of results
    """
    return get_monthly_prices(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_stk_limit(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
) -> dict:
    """Get limit-up and limit-down prices for a stock.
    
    A-share stocks have daily price limits (usually ±10%, ±20% for ChiNext/STAR).
    
    Args:
        ts_code: The stock ts_code (e.g., '000001.SZ')
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        limit: Maximum number of results
    """
    return get_stk_limit(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_suspend_d(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
) -> dict:
    """Get trading suspension and resumption events for a stock.
    
    Args:
        ts_code: The stock ts_code (e.g., '000001.SZ')
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        limit: Maximum number of results
    """
    return get_suspend_d(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_adj_factor(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
) -> dict:
    """Get adjustment factors for a stock (used to calculate adjusted prices).
    
    Args:
        ts_code: The stock ts_code (e.g., '000001.SZ')
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        limit: Maximum number of results
    """
    return get_adj_factor(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        store_dir=STORE_DIR,
    )


# -----------------------------------------------------------------------------
# Generic Query Tool
# -----------------------------------------------------------------------------


@tool
def tool_query_dataset(
    dataset: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
    order_by: str | None = None,
) -> dict:
    """Generic dataset query for advanced use cases.
    
    Available datasets:
    - daily, weekly, monthly: Price data
    - daily_basic: Valuation metrics
    - adj_factor: Adjustment factors
    - stk_limit: Price limits
    - suspend_d: Suspension events
    - stock_basic, stock_company: Company info
    - trade_cal: Trading calendar
    - new_share, namechange: IPO and name changes
    
    Args:
        dataset: Name of the dataset to query
        start_date: Start date filter (for trade-date datasets)
        end_date: End date filter (for trade-date datasets)
        limit: Maximum number of results
        order_by: Column to sort by
    """
    return query_dataset(
        dataset,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        order_by=order_by,
        store_dir=STORE_DIR,
    )


# -----------------------------------------------------------------------------
# Export all tools
# -----------------------------------------------------------------------------

ALL_TOOLS = [
    # Identity / Universe
    tool_resolve_symbol,
    tool_get_stock_basic,
    tool_get_stock_company,
    tool_get_universe,
    # Calendar
    tool_get_trading_days,
    tool_is_trading_day,
    tool_get_prev_trade_date,
    tool_get_next_trade_date,
    # IPO / Events
    tool_get_new_share,
    tool_get_namechange,
    # Market Data
    tool_get_daily_prices,
    tool_get_daily_adj_prices,
    tool_get_daily_basic,
    tool_get_weekly_prices,
    tool_get_monthly_prices,
    tool_get_stk_limit,
    tool_get_suspend_d,
    tool_get_adj_factor,
    # Generic
    tool_query_dataset,
]
