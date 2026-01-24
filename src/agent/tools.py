"""LangChain tool wrappers for the stock_data agent_tools.

Key improvements:
- Pagination support (offset/limit) for progressive navigation
- Clean responses with metadata (total_count, showing, has_more)
- Fuzzy search tool for finding stocks
"""

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
    get_stock_basic_detail,
    get_stock_company,
    get_stk_limit,
    get_suspend_d,
    get_trade_cal,
    get_trading_days,
    get_universe,
    get_weekly_prices,
    is_trading_day,
    list_industries,
    query_dataset,
    resolve_symbol,
    search_stocks,
)

# Get store directory from environment or use default
STORE_DIR = os.environ.get("STOCK_DATA_STORE_DIR", "../stock_data/store")


# -----------------------------------------------------------------------------
# Search & Discovery Tools (PRIMARY - use these first)
# -----------------------------------------------------------------------------


@tool
def tool_search_stocks(
    query: str,
    offset: int = 0,
    limit: int = 20,
) -> dict:
    """Search for stocks by name, symbol, code, or industry (fuzzy matching).
    
    This is the PRIMARY tool for finding stocks when user provides partial info.
    Use this first when user mentions a stock name, keyword, or partial code.
    
    Examples:
        - query="卫星" → finds stocks with "卫星" in name/industry
        - query="银行" → finds all bank stocks
        - query="300888" → finds stock by code
        - query="茅台" → finds 贵州茅台
    
    Args:
        query: Search term (Chinese name, symbol, ts_code, or industry keyword)
        offset: Skip first N results (for pagination, use to see more results)
        limit: Max results per page (default 20, max 100)
    
    Returns: {rows: [...], total_count: N, showing: "1-20", has_more: bool}
    """
    return search_stocks(query, offset=offset, limit=limit, store_dir=STORE_DIR)


@tool
def tool_list_industries() -> dict:
    """List all available industries in the stock database with stock counts.
    
    Use this to understand what industries exist before filtering.
    
    Returns: {industries: [...], count: N, stock_counts: {industry: count}}
    """
    return list_industries(store_dir=STORE_DIR)


@tool
def tool_resolve_symbol(symbol_or_ts_code: str) -> dict:
    """Resolve a stock symbol (e.g. '300888') or ts_code (e.g. '300888.SZ') to full info.
    
    Use after search to get the canonical ts_code format for other tools.
    
    Args:
        symbol_or_ts_code: Stock symbol or ts_code
    
    Returns: {symbol, ts_code, list_date}
    """
    return resolve_symbol(symbol_or_ts_code, store_dir=STORE_DIR)


# -----------------------------------------------------------------------------
# Stock Info Tools
# -----------------------------------------------------------------------------


@tool
def tool_get_stock_basic(
    ts_code: str | None = None,
    name_contains: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> dict:
    """Get stock basic info list with optional filters and pagination.
    
    For single stock detail, use tool_get_stock_basic_detail instead.
    
    Args:
        ts_code: Filter by exact ts_code (e.g., '000001.SZ')
        name_contains: Filter by name containing substring (e.g., '卫星')
        offset: Skip first N rows (for pagination)
        limit: Max rows (default 20, max 100)
    
    Returns: {rows: [...], total_count, showing, has_more}
    """
    return get_stock_basic(
        ts_code=ts_code,
        name_contains=name_contains,
        offset=offset,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_stock_basic_detail(ts_code: str) -> dict:
    """Get detailed stock basic info for a SINGLE stock (all available columns).
    
    Use this after finding the ts_code via search to get full details.
    
    Args:
        ts_code: The stock ts_code (e.g., '000001.SZ')
    
    Returns: {found: bool, data: {...all fields...}}
    """
    return get_stock_basic_detail(ts_code, store_dir=STORE_DIR)


@tool
def tool_get_stock_company(ts_code: str) -> dict:
    """Get detailed company profile (chairman, employees, main business, etc.).
    
    Args:
        ts_code: The stock ts_code (e.g., '000001.SZ')
    
    Returns: {found: bool, data: {...company info...}}
    """
    return get_stock_company(ts_code, store_dir=STORE_DIR)


@tool
def tool_get_universe(
    exchange: str | None = None,
    market: str | None = None,
    industry: str | None = None,
    area: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> dict:
    """Get filtered universe of listed stocks with pagination.
    
    Args:
        exchange: 'SSE' (Shanghai) or 'SZSE' (Shenzhen)
        market: '主板', '创业板', '科创板', 'CDR'
        industry: Industry name (use tool_list_industries to see options)
        area: Province (e.g., '北京', '广东')
        offset: Skip first N rows
        limit: Max rows (default 20, max 100)
    
    Returns: {rows: [ts_code, name, industry, market], total_count, showing, has_more}
    """
    return get_universe(
        exchange=exchange,
        market=market,
        industry=industry,
        area=area,
        offset=offset,
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
        start_date: Start date YYYYMMDD (e.g., '20240101')
        end_date: End date YYYYMMDD (e.g., '20240131')
    
    Returns: {trading_days: [...], count: N}
    """
    return get_trading_days(start_date, end_date, store_dir=STORE_DIR)


@tool
def tool_is_trading_day(date: str) -> dict:
    """Check if a specific date is a trading day.
    
    Args:
        date: Date YYYYMMDD (e.g., '20240115')
    
    Returns: {date, is_trading_day: bool}
    """
    return is_trading_day(date, store_dir=STORE_DIR)


@tool
def tool_get_prev_trade_date(date: str) -> dict:
    """Get the previous trading day before a given date.
    
    Args:
        date: Reference date YYYYMMDD
    
    Returns: {date, prev_trade_date}
    """
    return get_prev_trade_date(date, store_dir=STORE_DIR)


@tool
def tool_get_next_trade_date(date: str) -> dict:
    """Get the next trading day after a given date.
    
    Args:
        date: Reference date YYYYMMDD
    
    Returns: {date, next_trade_date}
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
    offset: int = 0,
    limit: int = 20,
) -> dict:
    """Get IPO / new share information with pagination.
    
    Args:
        year: Filter by IPO year (e.g., 2024)
        start_date: Filter IPOs from this date (YYYYMMDD)
        end_date: Filter IPOs until this date (YYYYMMDD)
        offset: Skip first N rows
        limit: Max rows (default 20)
    
    Returns: {rows: [...], total_count, showing, has_more}
    """
    return get_new_share(
        year=year,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_namechange(ts_code: str) -> dict:
    """Get stock name change history.
    
    Args:
        ts_code: The stock ts_code (e.g., '000001.SZ')
    
    Returns: {rows: [...], total_count, showing, has_more}
    """
    return get_namechange(ts_code, store_dir=STORE_DIR)


# -----------------------------------------------------------------------------
# Market Data Tools
# -----------------------------------------------------------------------------


@tool
def tool_get_daily_prices(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    offset: int = 0,
    limit: int = 30,
) -> dict:
    """Get daily OHLCV prices (unadjusted) for a stock.
    
    Returns most recent data first (descending by date).
    Default columns: trade_date, open, high, low, close, vol, pct_chg
    
    Args:
        ts_code: Stock ts_code (e.g., '000001.SZ')
        start_date: Start date YYYYMMDD (optional)
        end_date: End date YYYYMMDD (optional)
        offset: Skip first N rows (for pagination)
        limit: Max rows (default 30, max 200)
    
    Returns: {rows: [...], total_count, showing, has_more}
    """
    return get_daily_prices(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_daily_adj_prices(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    how: Literal["qfq", "hfq"] = "qfq",
    offset: int = 0,
    limit: int = 30,
) -> dict:
    """Get adjusted daily prices for a stock.
    
    - qfq (前复权): Forward-adjusted - prices adjusted to latest point
    - hfq (后复权): Backward-adjusted - prices adjusted from IPO
    
    Use qfq for recent trend analysis, hfq for historical comparison.
    
    Args:
        ts_code: Stock ts_code (e.g., '000001.SZ')
        start_date: Start date YYYYMMDD
        end_date: End date YYYYMMDD
        how: 'qfq' (forward-adjusted) or 'hfq' (backward-adjusted)
        offset: Skip first N rows
        limit: Max rows (default 30, max 200)
    
    Returns: {rows: [...], total_count, showing, has_more}
    """
    return get_daily_adj_prices(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        how=how,
        offset=offset,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_daily_basic(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    offset: int = 0,
    limit: int = 30,
) -> dict:
    """Get daily valuation metrics for a stock.
    
    Default columns: trade_date, pe_ttm, pb, total_mv, circ_mv, turnover_rate
    
    Args:
        ts_code: Stock ts_code (e.g., '000001.SZ')
        start_date: Start date YYYYMMDD
        end_date: End date YYYYMMDD
        offset: Skip first N rows
        limit: Max rows (default 30, max 200)
    
    Returns: {rows: [...], total_count, showing, has_more}
    """
    return get_daily_basic(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_weekly_prices(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> dict:
    """Get weekly OHLCV prices for a stock.
    
    Args:
        ts_code: Stock ts_code (e.g., '000001.SZ')
        start_date: Start date YYYYMMDD
        end_date: End date YYYYMMDD
        offset: Skip first N rows
        limit: Max rows (default 20, max 100)
    
    Returns: {rows: [...], total_count, showing, has_more}
    """
    return get_weekly_prices(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_monthly_prices(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    offset: int = 0,
    limit: int = 12,
) -> dict:
    """Get monthly OHLCV prices for a stock.
    
    Args:
        ts_code: Stock ts_code (e.g., '000001.SZ')
        start_date: Start date YYYYMMDD
        end_date: End date YYYYMMDD
        offset: Skip first N rows
        limit: Max rows (default 12, max 60)
    
    Returns: {rows: [...], total_count, showing, has_more}
    """
    return get_monthly_prices(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_stk_limit(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 30,
) -> dict:
    """Get limit-up and limit-down prices for a stock.
    
    A-share stocks have daily price limits (usually ±10%, ±20% for ChiNext/STAR).
    
    Args:
        ts_code: Stock ts_code (e.g., '000001.SZ')
        start_date: Start date YYYYMMDD
        end_date: End date YYYYMMDD
        limit: Max rows (default 30)
    
    Returns: {rows: [...], total_count, showing, has_more}
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
    limit: int = 30,
) -> dict:
    """Get trading suspension and resumption events for a stock.
    
    Args:
        ts_code: Stock ts_code (e.g., '000001.SZ')
        start_date: Start date YYYYMMDD
        end_date: End date YYYYMMDD
        limit: Max rows
    
    Returns: {rows: [...], total_count, showing, has_more}
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
    limit: int = 30,
) -> dict:
    """Get adjustment factors for a stock (used to calculate adjusted prices).
    
    Args:
        ts_code: Stock ts_code (e.g., '000001.SZ')
        start_date: Start date YYYYMMDD
        end_date: End date YYYYMMDD
        limit: Max rows
    
    Returns: {rows: [...], total_count, showing, has_more}
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
    offset: int = 0,
    limit: int = 50,
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
        dataset: Name of the dataset
        start_date: Start date filter (YYYYMMDD)
        end_date: End date filter (YYYYMMDD)
        offset: Skip first N rows
        limit: Max rows (default 50, max 200)
        order_by: Column to sort by (e.g., 'trade_date desc')
    
    Returns: {rows: [...], total_count, showing, has_more}
    """
    return query_dataset(
        dataset,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit,
        order_by=order_by,
        store_dir=STORE_DIR,
    )


# -----------------------------------------------------------------------------
# Export all tools
# -----------------------------------------------------------------------------

ALL_TOOLS = [
    # Search & Discovery (use these first!)
    tool_search_stocks,
    tool_list_industries,
    tool_resolve_symbol,
    # Stock Info
    tool_get_stock_basic,
    tool_get_stock_basic_detail,
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
