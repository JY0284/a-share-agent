"""LangChain tool wrappers for the A-Share agent.

Tool Philosophy:
- Discovery tools: Help agent find stocks and understand available data
- Execution tool: Let agent write Python code to analyze data directly

The agent should use discovery tools to find what it needs,
then use Python execution for actual data analysis and calculations.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

from langchain_core.tools import tool

from stock_data.runner import RunConfig
from stock_data.stats import fetch_stats_json
from stock_data.agent_tools import (
    get_daily_basic,
    get_daily_adj_prices,
    get_daily_prices,
    get_weekly_prices,
    get_monthly_prices,
    get_adj_factor,
    get_index_basic,
    get_index_daily_prices,
    get_fund_basic,
    get_etf_daily_prices,
    get_fund_nav,
    get_fund_share,
    get_fund_div,
    get_income,
    get_balancesheet,
    get_cashflow,
    get_forecast,
    get_express,
    get_dividend,
    get_fina_indicator,
    get_fina_audit,
    get_fina_mainbz,
    get_disclosure_date,
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

from agent.sandbox import clear_python_session, execute_python
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


# Category -> short Chinese label (for aggregated output)
_CATEGORY_ZH: dict[str, str] = {
    "market": "行情", "finance": "财报", "etf": "ETF", "basic": "基础", "macro": "宏观",
}


@tool
def tool_get_dataset_status() -> dict:
    """Get dataset coverage + freshness by category.

    Use when the user asks about data availability, latest dates, or when the agent needs
    to decide whether to warn about stale data (>2 trading days behind).

    The tool:
    - Computes the *current* Beijing date and current trading date (last trading day <= today)
    - Aggregates min/max partitions per dataset category
    - Computes trading-days-behind for each category
    - Appends a short freshness conclusion

    Backward compatible keys are preserved: `summary`, `latest_date`.

    Returns:
        {
          summary: str,
          latest_date: str|None,
          now: {date, date_compact, current_trade_date},
          categories: [{category, label_zh, min_date, latest_date, trading_days_behind, is_stale}],
          conclusion: str,
          report: str,
        }
    """
    import re

    from stock_data.datasets import dataset_info_map

    cfg = RunConfig(store_dir=STORE_DIR, rpm=500, workers=12)
    stats = fetch_stats_json(cfg, datasets="all")
    date_re = re.compile(r"(\d{8})")
    info_map = dataset_info_map()
    # Aggregate by category: {category: (min_date_iso, max_date_iso)}
    by_cat: dict[str, tuple[str | None, str | None]] = {}
    latest_ts: str | None = None

    tz = timezone(timedelta(hours=8))
    now_dt = datetime.now(tz)
    today_compact = now_dt.strftime("%Y%m%d")

    # Determine current trading date (last trading day <= today)
    try:
        is_td = is_trading_day(today_compact, store_dir=STORE_DIR)
        if bool(is_td.get("is_trading_day")):
            current_trade_date = today_compact
        else:
            prev = get_prev_trade_date(today_compact, store_dir=STORE_DIR)
            current_trade_date = prev.get("prev_trade_date") or today_compact
    except Exception:
        # Fallback: if trading calendar isn't available, use today's date.
        current_trade_date = today_compact

    def _fmt(val: str | None) -> str | None:
        if not val:
            return None
        m = date_re.search(val)
        if m:
            d = m[1]
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        return None

    def _to_compact(iso_date: str | None) -> str | None:
        if not iso_date:
            return None
        return iso_date.replace("-", "")

    def _trading_days_behind(latest_iso: str | None) -> int | None:
        """Return number of trading days from latest_iso to current_trade_date.

        0 means up-to-date at current_trade_date.
        1 means one trading day behind, etc.
        """
        latest_compact = _to_compact(latest_iso)
        if not latest_compact:
            return None
        if latest_compact >= current_trade_date:
            return 0
        try:
            nxt = get_next_trade_date(latest_compact, store_dir=STORE_DIR)
            start = nxt.get("next_trade_date")
            if not start:
                return None
            days = get_trading_days(start, current_trade_date, store_dir=STORE_DIR)
            td_list = days.get("trading_days") or []
            return int(len(td_list))
        except Exception:
            return None

    for s in stats:
        min_d = _fmt(s.get("min_partition"))
        max_d = _fmt(s.get("max_partition"))
        if not min_d and not max_d:
            continue
        cat = info_map.get(s["dataset"])
        key = cat.category if cat else "other"
        prev = by_cat.get(key, (None, None))
        pmin, pmax = prev
        if min_d and (pmin is None or min_d < pmin):
            pmin = min_d
        if max_d and (pmax is None or max_d > pmax):
            pmax = max_d
        by_cat[key] = (pmin, pmax)
        if max_d and (latest_ts is None or max_d > latest_ts):
            latest_ts = max_d

    parts = []
    for cat in ("market", "finance", "etf", "basic", "macro"):
        if cat not in by_cat:
            continue
        pmin, pmax = by_cat[cat]
        label = _CATEGORY_ZH.get(cat, cat)
        if pmin or pmax:
            parts.append(f"{label} {pmin or '-'}~{pmax or '-'}")
        else:
            parts.append(f"{label} 按代码")

    categories: list[dict] = []
    stale_labels: list[str] = []
    for cat in ("market", "finance", "etf", "basic", "macro"):
        if cat not in by_cat:
            continue
        pmin, pmax = by_cat[cat]
        label = _CATEGORY_ZH.get(cat, cat)
        behind = _trading_days_behind(pmax)
        is_stale = behind is not None and behind > 2
        if is_stale:
            stale_labels.append(f"{label}({behind}个交易日)")
        categories.append(
            {
                "category": cat,
                "label_zh": label,
                "min_date": pmin,
                "latest_date": pmax,
                "trading_days_behind": behind,
                "is_stale": is_stale,
            }
        )

    if stale_labels:
        conclusion = "⚠️ 数据可能滞后：" + "，".join(stale_labels) + "（>2个交易日）"
    else:
        conclusion = "✅ 数据新鲜：各类数据距今均不超过2个交易日（按交易日计）"

    now_info = {
        "date": now_dt.strftime("%Y-%m-%d"),
        "date_compact": today_compact,
        "current_trade_date": current_trade_date,
    }

    summary = "；".join(parts) + (f"；最新 {latest_ts}" if latest_ts else "")

    lines = [
        f"当前日期(北京时间)：{now_info['date']}（查询用：{now_info['date_compact']}）",
        f"当前交易日：{current_trade_date}",
        "数据最新日期（按类别）：",
    ]
    for row in categories:
        latest = row.get("latest_date") or "-"
        behind = row.get("trading_days_behind")
        behind_txt = "-" if behind is None else str(behind)
        warn = " ⚠️" if row.get("is_stale") else ""
        lines.append(f"- {row['label_zh']}: 最新 {latest}；落后 {behind_txt} 个交易日{warn}")
    lines.append("结论：" + conclusion)

    report = "\n".join(lines)

    return {
        "summary": summary,
        "latest_date": latest_ts,
        "now": now_info,
        "categories": categories,
        "conclusion": conclusion,
        "report": report,
    }


@tool
def tool_get_current_datetime() -> dict:
    """Get the actual current date and time in Beijing (UTC+8).

    Use this when you need to state the current system time/date to the user,
    or when assessing data freshness. The date in the system prompt header is
    set at server startup and may be stale if the server has been running for days.

    Returns: {date: YYYY-MM-DD, date_compact: YYYYMMDD, time: HH:MM, weekday: e.g. Wednesday, display_zh: Chinese display string}
    """
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz)
    weekdays_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekdays_zh = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    wd = now.weekday()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "date_compact": now.strftime("%Y%m%d"),
        "time": now.strftime("%H:%M"),
        "weekday": weekdays_en[wd],
        "weekday_zh": weekdays_zh[wd],
        "display_zh": f"{now.strftime('%Y年%m月%d日')}（{weekdays_zh[wd]}）{now.strftime('%H:%M')}（北京时间）",
    }


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


@tool
def tool_get_index_basic(
    ts_code: str | None = None,
    name_contains: str | None = None,
    market: str | None = None,
    publisher: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> dict:
    """List / search index basic info (指数基础信息) with pagination.

    Use this to discover index `ts_code` (e.g., 上证指数/沪深300/中证500).
    Then use `tool_get_index_daily_prices(ts_code, ...)` to fetch bars.
    """
    return get_index_basic(
        ts_code=ts_code,
        name_contains=name_contains,
        market=market,
        publisher=publisher,
        offset=offset,
        limit=min(limit or 20, 100),
        store_dir=STORE_DIR,
    )


@tool
def tool_get_fund_basic(
    ts_code: str | None = None,
    name_contains: str | None = None,
    management: str | None = None,
    fund_type: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> dict:
    """List / search ETF basic info (基金基础信息; 场内基金/ETF) with pagination.

    Use this to discover ETF `ts_code`, then use `tool_get_etf_daily_prices` / `tool_get_fund_nav` etc.
    """
    return get_fund_basic(
        ts_code=ts_code,
        name_contains=name_contains,
        management=management,
        fund_type=fund_type,
        offset=offset,
        limit=min(limit or 20, 100),
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
def tool_get_index_daily_prices(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> dict:
    """Get index daily bars (指数日线; most recent first)."""
    limit = _effective_limit(
        limit,
        start_date=start_date,
        end_date=end_date,
        default_recent=20,
        default_range=400,
        max_limit=500,
    )
    return get_index_daily_prices(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_etf_daily_prices(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> dict:
    """Get ETF daily bars (ETF日线; most recent first)."""
    limit = _effective_limit(
        limit,
        start_date=start_date,
        end_date=end_date,
        default_recent=20,
        default_range=400,
        max_limit=500,
    )
    return get_etf_daily_prices(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_fund_nav(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> dict:
    """Get ETF net asset value time series (单位净值; most recent first)."""
    limit = _effective_limit(
        limit,
        start_date=start_date,
        end_date=end_date,
        default_recent=30,
        default_range=500,
        max_limit=500,
    )
    return get_fund_nav(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_fund_share(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> dict:
    """Get ETF shares outstanding history (份额变动; most recent first)."""
    limit = _effective_limit(
        limit,
        start_date=start_date,
        end_date=end_date,
        default_recent=30,
        default_range=500,
        max_limit=500,
    )
    return get_fund_share(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=limit,
        store_dir=STORE_DIR,
    )


@tool
def tool_get_fund_div(ts_code: str, offset: int = 0, limit: int = 50) -> dict:
    """Get ETF dividend distribution history (分红送配)."""
    return get_fund_div(ts_code, offset=offset, limit=min(limit or 50, 200), store_dir=STORE_DIR)


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


@tool
def tool_get_income(
    ts_code: str,
    start_period: str | None = None,
    end_period: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> dict:
    """Get income statement (利润表) by report period (end_date), most recent first."""
    return get_income(
        ts_code,
        start_period=start_period,
        end_period=end_period,
        offset=offset,
        limit=min(limit or 20, 200),
        store_dir=STORE_DIR,
    )


@tool
def tool_get_balancesheet(
    ts_code: str,
    start_period: str | None = None,
    end_period: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> dict:
    """Get balance sheet (资产负债表) by report period (end_date), most recent first."""
    return get_balancesheet(
        ts_code,
        start_period=start_period,
        end_period=end_period,
        offset=offset,
        limit=min(limit or 20, 200),
        store_dir=STORE_DIR,
    )


@tool
def tool_get_cashflow(
    ts_code: str,
    start_period: str | None = None,
    end_period: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> dict:
    """Get cashflow statement (现金流量表) by report period (end_date), most recent first."""
    return get_cashflow(
        ts_code,
        start_period=start_period,
        end_period=end_period,
        offset=offset,
        limit=min(limit or 20, 200),
        store_dir=STORE_DIR,
    )


@tool
def tool_get_fina_indicator(
    ts_code: str,
    start_period: str | None = None,
    end_period: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> dict:
    """Get financial indicators (财务指标) by report period (end_date), most recent first."""
    return get_fina_indicator(
        ts_code,
        start_period=start_period,
        end_period=end_period,
        offset=offset,
        limit=min(limit or 20, 200),
        store_dir=STORE_DIR,
    )


@tool
def tool_get_forecast(
    ts_code: str,
    start_period: str | None = None,
    end_period: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict:
    """Get earnings forecast (业绩预告) by report period (end_date), most recent first."""
    return get_forecast(
        ts_code,
        start_period=start_period,
        end_period=end_period,
        offset=offset,
        limit=min(limit or 50, 200),
        store_dir=STORE_DIR,
    )


@tool
def tool_get_express(
    ts_code: str,
    start_period: str | None = None,
    end_period: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict:
    """Get earnings express (业绩快报) by report period (end_date), most recent first."""
    return get_express(
        ts_code,
        start_period=start_period,
        end_period=end_period,
        offset=offset,
        limit=min(limit or 50, 200),
        store_dir=STORE_DIR,
    )


@tool
def tool_get_dividend(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict:
    """Get dividend distribution history (分红送股), most recent first (best-effort)."""
    return get_dividend(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=min(limit or 50, 200),
        store_dir=STORE_DIR,
    )


@tool
def tool_get_fina_audit(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict:
    """Get audit opinions history (财务审计意见)."""
    return get_fina_audit(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=min(limit or 50, 200),
        store_dir=STORE_DIR,
    )


@tool
def tool_get_fina_mainbz(
    ts_code: str,
    start_period: str | None = None,
    end_period: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict:
    """Get main business composition (主营业务构成) by report period (end_date), most recent first."""
    return get_fina_mainbz(
        ts_code,
        start_period=start_period,
        end_period=end_period,
        offset=offset,
        limit=min(limit or 50, 200),
        store_dir=STORE_DIR,
    )


@tool
def tool_get_disclosure_date(
    ts_code: str | None = None,
    end_date: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict:
    """Get financial report disclosure schedule (财报披露日期表) with pagination."""
    return get_disclosure_date(
        ts_code=ts_code,
        end_date=end_date,
        offset=offset,
        limit=min(limit or 50, 200),
        store_dir=STORE_DIR,
    )


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
    """Execute Python code for CALCULATIONS that other tools CANNOT do.
    
    ⚠️ THIS IS A LAST RESORT TOOL - only use when other tools cannot answer!
    
    - "最近股价" → use tool_get_daily_prices, NOT Python
    - "最近一个月股价" → use tool_get_daily_prices(start_date=...), NOT Python
    - "PE是多少" → use tool_get_daily_basic, NOT Python
    - "公司主营业务" → use tool_get_stock_company, NOT Python
    
    Only use Python when you need to COMPUTE (rolling, groupby, compare, rank, etc.)!
    
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
    
    # Index data (指数)
    idx = store.read("index_basic")  # discover index codes
    df = store.index_daily("000300.SH", start_date="20230101")  # HS300 example
    
    # ETF / fund data
    fb = store.read("fund_basic")  # discover ETF codes
    etf_px = store.read("etf_daily", where={"ts_code": "510300.SH"}, start_date="20230101")
    nav = store.fund_nav("510300.SH", start_date="20230101")
    share = store.fund_share("510300.SH", start_date="20230101")
    div = store.fund_div("510300.SH")
    
    # Valuation data
    df = store.daily_basic(ts_code)  # pe_ttm, pb, total_mv, circ_mv
    # ⚠️ IMPORTANT: store.daily_basic() does NOT accept 'limit' parameter!
    # If you need to limit rows, use: df.tail(n) or df.head(n) after loading
    
    # Finance statements (report-period end_date, not trading days)
    inc = store.income(ts_code, start_period="20200101")
    bs = store.balancesheet(ts_code, start_period="20200101")
    cf = store.cashflow(ts_code, start_period="20200101")
    fi = store.fina_indicator(ts_code, start_period="20200101")
    
    # Other
    df = store.stock_basic(ts_code=ts_code)
    df = store.stock_company(ts_code=ts_code)
    days = store.trading_days(start_date, end_date)
    ```
    
    ## ⚠️ CRITICAL: Tool API vs Store API
    **DO NOT confuse tool parameters with store method parameters!**
    - `tool_get_daily_basic(ts_code, limit=10)` ← Tool accepts 'limit'
    - `store.daily_basic(ts_code)` ← Store method does NOT accept 'limit'
    
    If you need to limit rows when using store, do it AFTER loading:
    ```python
    # ✅ CORRECT
    df = store.daily_basic(ts_code).tail(10)  # Get last 10 rows
    
    # ❌ WRONG - will cause error!
    df = store.daily_basic(ts_code, limit=10)  # ERROR: unexpected keyword argument 'limit'
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
    
    ## Session state
    Variables (DataFrames, lists, etc.) persist across multiple tool_execute_python
    calls in the same agent thread. You can load data once, then run follow-up
    calculations in a later call using the same names (e.g. `df`, `result`).
    
    Args:
        code: Python code that uses `store` to load and analyze data.
        skills_used: List of skill IDs that guided this code (from tool_search_skills/tool_load_skill).
    
    Returns:
        {"success": bool, "output": str, "error": str|None, "result": str, "skills_used": list}
    """
    out = execute_python(code)
    out["skills_used"] = skills_used or []
    return out


@tool
def tool_clear_python_session() -> dict:
    """Clear the Python execution session state for this thread.
    
    Call this when you want to start fresh (e.g. user asks a new unrelated question)
    so that old variables/DataFrames no longer persist.
    
    Returns: {"cleared": True}
    """
    clear_python_session()
    return {"cleared": True}


# =============================================================================
# EXPORT
# =============================================================================

ALL_TOOLS = [
    # Discovery (use these first!)
    tool_search_stocks,
    tool_get_dataset_status,
    tool_get_current_datetime,
    tool_list_industries,
    tool_resolve_symbol,
    tool_get_stock_basic_detail,
    tool_get_stock_company,
    tool_get_universe,
    tool_get_index_basic,
    tool_get_fund_basic,
    # Simple Data (for basic lookups, no calculation)
    tool_get_daily_prices,
    tool_get_daily_adj_prices,
    tool_get_daily_basic,
    tool_get_weekly_prices,
    tool_get_monthly_prices,
    tool_get_index_daily_prices,
    tool_get_etf_daily_prices,
    tool_get_fund_nav,
    tool_get_fund_share,
    tool_get_fund_div,
    tool_get_adj_factor,
    tool_get_stk_limit,
    tool_get_suspend_d,
    tool_get_new_share,
    tool_get_namechange,
    tool_get_income,
    tool_get_balancesheet,
    tool_get_cashflow,
    tool_get_fina_indicator,
    tool_get_forecast,
    tool_get_express,
    tool_get_dividend,
    tool_get_fina_audit,
    tool_get_fina_mainbz,
    tool_get_disclosure_date,
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
    tool_clear_python_session,
]
