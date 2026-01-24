"""LangChain tool wrappers for the A-Share agent.

Tool Philosophy:
- Discovery tools: Help agent find stocks and understand available data
- Execution tool: Let agent write Python code to analyze data directly

The agent should use discovery tools to find what it needs,
then use Python execution for actual data analysis and calculations.
"""

from __future__ import annotations

import os

from langchain_core.tools import tool

from stock_data.agent_tools import (
    get_stock_basic_detail,
    get_stock_company,
    get_next_trade_date,
    get_prev_trade_date,
    get_trading_days,
    is_trading_day,
    list_industries,
    resolve_symbol,
    search_stocks,
)

from agent.sandbox import execute_python
from agent.skills import list_skills, load_skill, search_skills

# Get store directory from environment or use default
STORE_DIR = os.environ.get("STOCK_DATA_STORE_DIR", "../stock_data/store")


# =============================================================================
# DISCOVERY TOOLS - For finding stocks and understanding the data landscape
# =============================================================================


@tool
def tool_search_stocks(
    query: str,
    offset: int = 0,
    limit: int = 20,
) -> dict:
    """Search for stocks by name, symbol, code, or industry (fuzzy matching).
    
    This is the PRIMARY tool for finding stocks. Use this FIRST when user 
    mentions any stock name, keyword, or partial code.
    
    Examples:
        - query="卫星" → finds stocks with "卫星" in name/industry
        - query="银行" → finds all bank stocks
        - query="300888" → finds stock by code
        - query="茅台" → finds 贵州茅台
    
    Args:
        query: Search term (Chinese name, symbol, ts_code, or industry keyword)
        offset: Skip first N results (for pagination)
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
    
    Use after search to get the canonical ts_code format.
    
    Args:
        symbol_or_ts_code: Stock symbol or ts_code
    
    Returns: {symbol, ts_code, list_date}
    """
    return resolve_symbol(symbol_or_ts_code, store_dir=STORE_DIR)


@tool
def tool_get_stock_basic_detail(ts_code: str) -> dict:
    """Get detailed stock basic info for a SINGLE stock (all available fields).
    
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
def tool_list_skills() -> dict:
    """List available agent skills from a-share-agent/skills/*/experience.md.

    Returns a lightweight list: name, description, tags, and skill_id (directory name).
    """
    skills = list_skills()
    rows = []
    for s in skills:
        rows.append(
            {
                "skill_id": s.path.parent.name,
                "name": s.name,
                "description": s.description,
                "tags": s.tags,
            }
        )
    return {"rows": rows, "count": len(rows)}


@tool
def tool_search_skills(query: str, limit: int = 5) -> dict:
    """Search skills relevant to a query (used before Python execution).

    Args:
        query: User question or subtask
        limit: Max skills to return (recommend 1-3 for injection)
    """
    hits = search_skills(query, k=limit)
    rows = []
    for s in hits:
        rows.append(
            {
                "skill_id": s.path.parent.name,
                "name": s.name,
                "description": s.description,
                "tags": s.tags,
            }
        )
    return {"query": query, "rows": rows, "count": len(rows)}


@tool
def tool_load_skill(skill_id: str) -> dict:
    """Load full skill content (frontmatter + markdown body)."""
    return load_skill(skill_id)


# =============================================================================
# CALENDAR TOOLS - For trading day queries
# =============================================================================


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


# =============================================================================
# PYTHON EXECUTION TOOL - For data analysis and calculations
# =============================================================================


@tool
def tool_execute_python(code: str, skills_used: list[str] | None = None) -> dict:
    """Execute Python code for data analysis and calculations.
    
    This is the MAIN tool for analyzing stock data. You have full access to:
    
    ## Pre-loaded Libraries
    - `pd` / `pandas`: Data manipulation
    - `np` / `numpy`: Numerical computation  
    - `plt` / `matplotlib.pyplot`: Plotting (if needed)
    
    ## Stock Data Access
    - `store`: StockStore instance with these methods:
    
    ### Price Data (returns pandas DataFrame)
    ```python
    # Daily prices (OHLCV + pct_chg)
    df = store.daily(ts_code, start_date=None, end_date=None)
    
    # Adjusted prices (qfq=forward, hfq=backward)
    df = store.daily_adj(ts_code, start_date=None, end_date=None, how="qfq")
    
    # Weekly/monthly prices
    df = store.weekly(ts_code, start_date=None, end_date=None)
    df = store.monthly(ts_code, start_date=None, end_date=None)
    ```
    
    ### Valuation Data
    ```python
    # Daily metrics: pe_ttm, pb, total_mv, circ_mv, turnover_rate, etc.
    df = store.daily_basic(ts_code, start_date=None, end_date=None)
    ```
    
    ### Other Data
    ```python
    # Stock basic info
    df = store.stock_basic(ts_code=None, symbol=None)
    
    # Company profile  
    df = store.stock_company(ts_code=ts_code)
    
    # Trading calendar
    days = store.trading_days(start_date, end_date)
    
    # Adjustment factors
    df = store.adj_factor(ts_code, start_date=None, end_date=None)
    
    # Suspension events
    df = store.suspend_d(ts_code, start_date=None, end_date=None)
    
    # Limit up/down prices
    df = store.stk_limit(ts_code, start_date=None, end_date=None)
    ```
    
    ## Tips
    1. Use `print()` to show intermediate results
    2. Assign final result to `result` variable for clean output
    3. DataFrames are sorted by trade_date ascending by default
    4. Date format is YYYYMMDD string (e.g., "20240101")
    5. ts_code format is "symbol.exchange" (e.g., "600519.SH", "000001.SZ")
    
    ## Example: Calculate 20-day moving average
    ```python
    df = store.daily("600519.SH", start_date="20240101")
    df = df.sort_values("trade_date")
    df["ma20"] = df["close"].rolling(20).mean()
    result = df[["trade_date", "close", "ma20"]].tail(10)
    print(result)
    ```
    
    ## Example: Compare PE ratios of multiple stocks
    ```python
    stocks = ["600519.SH", "000858.SZ", "000568.SZ"]  # 白酒三巨头
    results = []
    for ts_code in stocks:
        basic = store.daily_basic(ts_code).tail(1)
        if not basic.empty:
            results.append({
                "ts_code": ts_code,
                "pe_ttm": basic["pe_ttm"].values[0],
                "pb": basic["pb"].values[0],
                "total_mv": basic["total_mv"].values[0] / 10000  # 亿元
            })
    result = pd.DataFrame(results)
    print(result)
    ```
    
    Args:
        code: Python code to execute. Use print() for output, assign to `result` for return value.
    
    Returns:
        {
            "success": bool,
            "output": str,       # stdout from print()
            "error": str | None, # error message if failed
            "result": str,       # formatted result value
            "skills_used": list[str]  # names/ids of skills used to produce this code
        }
    """
    out = execute_python(code)
    out["skills_used"] = skills_used or []
    return out


# =============================================================================
# EXPORT
# =============================================================================

ALL_TOOLS = [
    # Discovery (use these first!)
    tool_search_stocks,
    tool_list_industries,
    tool_list_skills,
    tool_search_skills,
    tool_load_skill,
    tool_resolve_symbol,
    tool_get_stock_basic_detail,
    tool_get_stock_company,
    # Calendar
    tool_get_trading_days,
    tool_is_trading_day,
    tool_get_prev_trade_date,
    tool_get_next_trade_date,
    # Python Execution (main analysis tool)
    tool_execute_python,
]
