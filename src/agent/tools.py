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
    get_daily_basic,
    get_daily_adj_prices,
    get_daily_prices,
    get_weekly_prices,
    get_monthly_prices,
    get_adj_factor,
    get_stock_basic_detail,
    get_stock_company,
    get_universe,
    get_next_trade_date,
    get_prev_trade_date,
    get_trading_days,
    is_trading_day,
    get_stk_limit,
    get_suspend_d,
    get_new_share,
    get_namechange,
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
    """Search for stocks by name, symbol, code, or industry keyword (fuzzy matching).

    This is the PRIMARY entry point when the user mentions a stock name / keyword / partial code.

    **Related-stocks workflow (important):**
    - 1) Use this tool to get a few *seed* matches (e.g., the target company + close variants).
    - 2) If the user asks for "相关/同概念/同类" stocks, DON'T stop here:
        - Use `tool_list_industries` to discover the most relevant 行业 keywords.
        - Then use `tool_get_universe(industry=...)` (optionally market/area/exchange filters) to list more
          related stocks in that 行业, with pagination.
        - Use `tool_get_stock_company` / `tool_get_stock_basic_detail` to verify business lines when the
          theme is ambiguous (e.g., “卫星/航天/军工”, “AI/算力/半导体”).

    Examples:
        - query="卫星" → finds stocks with "卫星" in name/industry (good for seeds)
        - query="银行" → finds bank stocks; for a complete list, prefer `tool_get_universe(industry="银行")`
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
def tool_get_universe(
    exchange: str | None = None,
    market: str | None = None,
    industry: str | None = None,
    area: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> dict:
    """Get filtered universe of listed stocks with pagination.
    
    Use for queries like "列出银行股" or "深圳创业板有哪些股票".
    
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


# =============================================================================
# SIMPLE DATA TOOLS - For basic price/valuation lookups (no calculation needed)
# =============================================================================


def _effective_limit(
    limit: int | None,
    *,
    start_date: str | None,
    end_date: str | None,
    default_recent: int,
    default_range: int,
    max_limit: int,
) -> int:
    """Choose a reasonable page size for agent-facing tools.

    - If the caller explicitly sets `limit`, respect it (clamped to max_limit).
    - If a date range is provided, default to a larger page size.
    - Otherwise default to a small "recent rows" lookup size.
    """
    if limit is None:
        limit = default_range if (start_date or end_date) else default_recent
    limit = int(limit)
    if limit <= 0:
        limit = default_recent
    return min(limit, max_limit)


@tool
def tool_get_daily_prices(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> dict:
    """Get daily prices for a stock (supports date range + pagination).
    
    Use for simple queries like "获取茅台最近的股价" or "最近5天收盘价".
    For complex analysis (MA, trends, comparisons), use tool_execute_python instead.
    
    Returns most recent data first (descending by date).
    Columns: trade_date, open, high, low, close, vol, pct_chg
    
    Args:
        ts_code: Stock ts_code (e.g., '600519.SH')
        start_date: Start date YYYYMMDD (optional)
        end_date: End date YYYYMMDD (optional)
        offset: Skip first N rows (for pagination)
        limit: Max rows. If omitted and a date range is provided, returns a larger page.
    
    Returns: {rows: [...], total_count, showing, has_more}
    """
    limit = _effective_limit(
        limit,
        start_date=start_date,
        end_date=end_date,
        default_recent=10,
        default_range=200,
        max_limit=200,
    )
    return get_daily_prices(
        ts_code,
        start_date=start_date,
        end_date=end_date,
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
    limit: int | None = None,
) -> dict:
    """Get daily valuation metrics for a stock (supports date range + pagination).
    
    Use for simple queries like "茅台的PE是多少" or "获取最新估值数据".
    For complex analysis (PE comparisons, trend), use tool_execute_python instead.
    
    Columns: trade_date, pe_ttm, pb, total_mv, circ_mv, turnover_rate
    
    Args:
        ts_code: Stock ts_code (e.g., '600519.SH')
        start_date: Start date YYYYMMDD
        end_date: End date YYYYMMDD
        offset: Skip first N rows (for pagination)
        limit: Max rows. If omitted and a date range is provided, returns a larger page.
    
    Returns: {rows: [...], total_count, showing, has_more}
    """
    limit = _effective_limit(
        limit,
        start_date=start_date,
        end_date=end_date,
        default_recent=10,
        default_range=200,
        max_limit=200,
    )
    return get_daily_basic(
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
    how: str = "qfq",
    offset: int = 0,
    limit: int | None = None,
) -> dict:
    """Get adjusted daily prices (supports date range + pagination).

    Args:
        ts_code: Stock ts_code (e.g., '600519.SH')
        start_date: Start date YYYYMMDD (optional)
        end_date: End date YYYYMMDD (optional)
        how: 'qfq' (forward), 'hfq' (backward), or 'both'
        offset: Skip first N rows (for pagination)
        limit: Max rows. If omitted and a date range is provided, returns a larger page.
    """
    limit = _effective_limit(
        limit,
        start_date=start_date,
        end_date=end_date,
        default_recent=30,
        default_range=200,
        max_limit=200,
    )
    return get_daily_adj_prices(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        how=how,  # type: ignore[arg-type]
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
    limit: int | None = None,
) -> dict:
    """Get weekly prices (supports date range + pagination)."""
    limit = _effective_limit(
        limit,
        start_date=start_date,
        end_date=end_date,
        default_recent=20,
        default_range=100,
        max_limit=100,
    )
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
    limit: int | None = None,
) -> dict:
    """Get monthly prices (supports date range + pagination)."""
    limit = _effective_limit(
        limit,
        start_date=start_date,
        end_date=end_date,
        default_recent=12,
        default_range=60,
        max_limit=60,
    )
    return get_monthly_prices(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_adj_factor(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> dict:
    """Get adjustment factors (supports date range + pagination)."""
    limit = _effective_limit(
        limit,
        start_date=start_date,
        end_date=end_date,
        default_recent=30,
        default_range=200,
        max_limit=200,
    )
    return get_adj_factor(
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
    offset: int = 0,
    limit: int | None = None,
) -> dict:
    """Get limit-up/limit-down prices (supports date range + pagination)."""
    limit = _effective_limit(
        limit,
        start_date=start_date,
        end_date=end_date,
        default_recent=30,
        default_range=200,
        max_limit=200,
    )
    return get_stk_limit(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_suspend_d(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> dict:
    """Get suspension/resumption events (supports date range + pagination)."""
    limit = _effective_limit(
        limit,
        start_date=start_date,
        end_date=end_date,
        default_recent=30,
        default_range=100,
        max_limit=100,
    )
    return get_suspend_d(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_new_share(
    year: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    ts_code: str | None = None,
    symbol_or_sub_code: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> dict:
    """Get IPO / new share info (supports date range + pagination)."""
    # Here "date range" is the common query path, so default to a larger page.
    limit = _effective_limit(
        limit,
        start_date=start_date,
        end_date=end_date,
        default_recent=20,
        default_range=100,
        max_limit=100,
    )
    return get_new_share(
        year=year,
        start_date=start_date,
        end_date=end_date,
        ts_code=ts_code,
        symbol_or_sub_code=symbol_or_sub_code,
        offset=offset,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_namechange(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Get name change history (supports date range filtering)."""
    return get_namechange(ts_code, start_date=start_date, end_date=end_date, store_dir=STORE_DIR)


# =============================================================================
# SKILLS TOOLS - For discovering and loading agent skills
# =============================================================================


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
    """Execute Python code for DATA ANALYSIS and CALCULATIONS only.
    
    ⚠️ IMPORTANT: This tool is for COMPUTING with data, NOT for generating text!
    
    ## FORBIDDEN - Do NOT write code like this:
    ```python
    # ❌ WRONG: Print-only code without using store
    print("=== 分析报告 ===")
    print("1. 公司主营业务：xxx")
    print("2. 竞争优势：xxx")
    ```
    This is a MISUSE of the Python tool. If you want to explain something,
    just respond in text directly - don't wrap it in print statements!
    
    ## REQUIRED - Your code MUST:
    1. Use `store` to load actual data from the database
    2. Perform real calculations (rolling, groupby, comparisons, etc.)
    3. Output computed results, not hand-written text
    
    ## Before Using This Tool
    1. First call `tool_search_skills(query)` to find relevant coding patterns
    2. Load useful skills with `tool_load_skill(skill_id)`
    3. Apply skill guidance in your code
    4. Pass `skills_used=[skill_id, ...]` when calling this tool
    
    ## Pre-loaded Libraries
    - `pd` / `pandas`: Data manipulation
    - `np` / `numpy`: Numerical computation  
    - `plt` / `matplotlib.pyplot`: Plotting (if needed)
    
    ## Stock Data Access via `store`
    ```python
    # Price data
    df = store.daily(ts_code, start_date=None, end_date=None)
    df = store.daily_adj(ts_code, how="qfq")  # Adjusted prices
    df = store.weekly(ts_code)
    df = store.monthly(ts_code)
    
    # Valuation data
    df = store.daily_basic(ts_code)  # pe_ttm, pb, total_mv, circ_mv
    
    # Other
    df = store.stock_basic(ts_code=ts_code)
    df = store.stock_company(ts_code=ts_code)
    days = store.trading_days(start_date, end_date)
    ```
    
    ## Good Example: Calculate MA and find golden cross
    ```python
    df = store.daily("600519.SH", start_date="20240101")
    df = df.sort_values("trade_date")
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["golden_cross"] = (df["ma5"] > df["ma20"]) & (df["ma5"].shift(1) <= df["ma20"].shift(1))
    result = df[df["golden_cross"]][["trade_date", "close", "ma5", "ma20"]]
    print(result)
    ```
    
    ## Good Example: Compare PE ratios
    ```python
    stocks = ["600519.SH", "000858.SZ", "000568.SZ"]
    results = []
    for ts_code in stocks:
        basic = store.daily_basic(ts_code).tail(1)
        if not basic.empty:
            results.append({
                "ts_code": ts_code,
                "pe_ttm": basic["pe_ttm"].values[0],
                "total_mv(亿)": basic["total_mv"].values[0] / 10000
            })
    result = pd.DataFrame(results)
    print(result)
    ```
    
    Args:
        code: Python code that uses `store` to load and analyze data.
        skills_used: List of skill IDs that guided this code (from tool_search_skills/tool_load_skill).
    
    Returns:
        {"success": bool, "output": str, "error": str|None, "result": str, "skills_used": list}
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
    tool_resolve_symbol,
    tool_get_stock_basic_detail,
    tool_get_stock_company,
    tool_get_universe,
    # Simple Data (for basic lookups, no calculation)
    tool_get_daily_prices,
    tool_get_daily_adj_prices,
    tool_get_daily_basic,
    tool_get_weekly_prices,
    tool_get_monthly_prices,
    tool_get_adj_factor,
    tool_get_stk_limit,
    tool_get_suspend_d,
    tool_get_new_share,
    tool_get_namechange,
    # Calendar
    tool_get_trading_days,
    tool_is_trading_day,
    tool_get_prev_trade_date,
    tool_get_next_trade_date,
    # Skills (for Python execution)
    tool_list_skills,
    tool_search_skills,
    tool_load_skill,
    # Python Execution (for complex analysis only)
    tool_execute_python,
]
