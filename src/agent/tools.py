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
    # Market Extras
    get_moneyflow,
    get_fx_daily,
    # Macro
    get_lpr,
    get_cpi,
    get_cn_sf,
    get_cn_m,
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
# MARKET EXTRAS TOOLS - Money flow, FX (资金流向/外汇)
# =============================================================================


@tool
def tool_get_moneyflow(
    ts_code: str | None = None,
    trade_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict:
    """Get A-share stock money flow data (资金流向).

    ⚠️ Requires at least one filter: ts_code, trade_date, or start_date/end_date.

    Date format: YYYYMMDD.

    Args:
        ts_code: Stock ts_code (e.g. '600519.SH')
        trade_date: Specific trading date
        start_date: Start of range
        end_date: End of range
        offset: Skip first N rows
        limit: Max rows (default 50, max 500)

    Returns: {rows: [trade_date, ts_code, buy_sm_vol, sell_sm_vol, ...], ...}
    """
    return get_moneyflow(
        ts_code=ts_code,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=min(limit or 50, 500),
        store_dir=STORE_DIR,
    )


@tool
def tool_get_fx_daily(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict:
    """Get FX daily quotes for a currency pair (外汇日线).

    Date format: YYYYMMDD (GMT in upstream).

    Args:
        ts_code: Currency pair code (e.g. 'USDCNH' for USD/CNH)
        start_date: Start date YYYYMMDD
        end_date: End date YYYYMMDD
        offset: Skip first N rows
        limit: Max rows (default 50, max 500)

    Returns: {rows: [trade_date, open, high, low, close, ...], ...}
    """
    return get_fx_daily(
        ts_code,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=min(limit or 50, 500),
        store_dir=STORE_DIR,
    )


# =============================================================================
# MACRO TOOLS - LPR, CPI, Social Financing, Money Supply (宏观数据)
# =============================================================================


@tool
def tool_get_lpr(
    date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict:
    """Get LPR (贷款市场报价利率) history.

    Date format: YYYYMMDD.

    Args:
        date: Specific date
        start_date: Start of range
        end_date: End of range
        offset: Skip first N rows
        limit: Max rows (default 50, max 200)

    Returns: {rows: [date, lpr_1y, lpr_5y, ...], ...}
    """
    return get_lpr(
        date=date,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        limit=min(limit or 50, 200),
        store_dir=STORE_DIR,
    )


@tool
def tool_get_cpi(
    month: str | None = None,
    start_month: str | None = None,
    end_month: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict:
    """Get CPI (居民消费价格指数) history.

    Month format: YYYYMM (e.g. '202601').

    Args:
        month: Specific month
        start_month: Start of range
        end_month: End of range
        offset: Skip first N rows
        limit: Max rows (default 50, max 200)

    Returns: {rows: [month, nt_yoy, nt_mom, ...], ...}
    """
    return get_cpi(
        month=month,
        start_month=start_month,
        end_month=end_month,
        offset=offset,
        limit=min(limit or 50, 200),
        store_dir=STORE_DIR,
    )


@tool
def tool_get_cn_sf(
    month: str | None = None,
    start_month: str | None = None,
    end_month: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict:
    """Get CN social financing (社融) monthly series.

    Month format: YYYYMM (e.g. '202601').

    Args:
        month: Specific month
        start_month: Start of range
        end_month: End of range
        offset: Skip first N rows
        limit: Max rows (default 50, max 200)

    Returns: {rows: [month, ...social financing components...], ...}
    """
    return get_cn_sf(
        month=month,
        start_month=start_month,
        end_month=end_month,
        offset=offset,
        limit=min(limit or 50, 200),
        store_dir=STORE_DIR,
    )


@tool
def tool_get_cn_m(
    month: str | None = None,
    start_month: str | None = None,
    end_month: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict:
    """Get CN money supply (货币供应量) monthly series (M0/M1/M2).

    Month format: YYYYMM (e.g. '202601').

    Args:
        month: Specific month
        start_month: Start of range
        end_month: End of range
        offset: Skip first N rows
        limit: Max rows (default 50, max 200)

    Returns: {rows: [month, m0, m1, m2, m0_yoy, m1_yoy, m2_yoy, ...], ...}
    """
    return get_cn_m(
        month=month,
        start_month=start_month,
        end_month=end_month,
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


@tool(response_format="content_and_artifact")
def tool_execute_python(code: str, skills_used: list[str] | None = None) -> tuple[dict, dict]:
    """Execute Python code for CALCULATIONS that other tools CANNOT do.
    
    ⚠️ THIS IS A LAST RESORT TOOL - only use when other tools cannot answer!
    
    - "茅台最近股价" → use tool_stock_snapshot("茅台"), NOT Python
    - "最近一个月股价" → use tool_stock_snapshot then tool_get_daily_prices(start_date=...), NOT Python
    - "PE是多少" → use tool_stock_snapshot (includes valuation), NOT Python
    - "公司主营业务" → use tool_stock_snapshot (includes company info), NOT Python
    - "同行对比" → use tool_peer_comparison(ts_code), NOT Python
    
    Only use Python when you need to COMPUTE (rolling, groupby, compare, rank, etc.)!
    
    ## Skills - ALWAYS use tool_search_and_load_skill first!
    Before writing Python code, use `tool_search_and_load_skill(query_or_skill_id)` to:
    1. Show users which skill patterns you're following (transparency)
    2. Get detailed code examples and best practices
    
    Available skills: rolling_indicators, backtest_ma_crossover, backtest_macd,
    statistical_analysis, time_series_forecast, risk_metrics, etc.
    
    Use `tool_search_and_load_skill(query)` to find and load relevant skills in one call.
    
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
    
    ## Pre-loaded Libraries
    - `pd` / `pandas`: Data manipulation
    - `np` / `numpy`: Numerical computation  
    - `sm` / `statsmodels.api`: Statistical analysis (OLS, time series)
    - `arch_model`: GARCH volatility models
    - `plt` / `matplotlib.pyplot`: Plotting/charting
    
    ## 📊 Creating Charts (Automatic Figure Capture)
    
    Matplotlib figures are **automatically captured** and saved to persistent storage.
    Each figure gets a unique ID and URL for frontend display.
    
    **When to create charts:**
    - Backtests: equity curves, drawdown charts
    - Comparisons: multi-stock performance overlay
    - Trends: price with MA/Bollinger/MACD indicators
    - Analysis: correlation heatmaps, return distributions
    
    **Rules for plotting:**
    1. Set a descriptive `plt.title()` - it appears as the chart caption
    2. Use `plt.tight_layout()` before finishing
    3. Do NOT call `plt.show()` or `plt.savefig()` - figures are captured automatically
    4. For Chinese text, set font at start:
       ```python
       plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
       plt.rcParams["axes.unicode_minus"] = False
       ```
    
    **Referencing figures in your response:**
    After execution, the tool result includes `generated_figures` list with each figure's reference.
    **COPY the `reference` string into your response** - the frontend renders it as a clickable thumbnail!
    
    Tool result example:
    ```json
    "generated_figures": [{"id": "fig_9a8b7c", "title": "策略回测", "reference": "[[fig:fig_9a8b7c|策略回测]]"}]
    ```
    
    Your response should include the reference from the result:
    ```
    回测结果如下：
    
    [[fig:fig_9a8b7c|策略回测]]
    
    从图中可以看出，策略年化收益率为...
    ```
    
    **Backtest example with equity curve:**
    ```python
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    
    # ... backtest logic producing df with 'equity' column ...
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(df["trade_date"], df["equity"], label="策略净值")
    ax1.set_title("MA双均线策略回测 - 600519.SH")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.fill_between(df["trade_date"], df["drawdown"], 0, alpha=0.5, color="red")
    ax2.set_title("回撤")
    ax2.set_ylabel("回撤 %")
    plt.tight_layout()
    ```
    
    Load `tool_search_and_load_skill("plotting_charts")` for more chart patterns.
    
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
    
    # Market extras
    mf = store.read("moneyflow", where={"ts_code": ts_code}, start_date="20240101")  # Money flow
    fx = store.read("fx_daily", where={"ts_code": "USDCNH"}, start_date="20240101")  # FX rates
    
    # Macro data (month format YYYYMM)
    lpr = store.read("lpr")              # LPR rates
    cpi = store.read("cpi")              # CPI data
    sf = store.read("cn_sf")             # Social financing
    m = store.read("cn_m")               # Money supply M0/M1/M2
    
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
        skills_used: List of skill IDs loaded via tool_search_and_load_skill before this call.
    
    Returns:
        Tuple of (content, artifact):
        - content: {"success": bool, "output": str, "error": str|None, "result": str, "skills_used": list, "skill_hint": str|None}
          This is sent to the LLM as the tool result.
        - artifact: {"figures": list|None}
          This is stored in ToolMessage.artifact for frontend display, NOT sent to LLM.
          figures: [{"image": "<base64>", "title": "Figure title", "format": "png"}, ...]
    """
    from agent.skills import smart_select_skills
    
    out = execute_python(code)
    out["skills_used"] = skills_used or []
    out["skill_hint"] = None
    
    # If no skills were explicitly used, detect relevant skills and add a hint
    if not skills_used:
        skill_selection = smart_select_skills(code=code, query="", max_skills=2)
        recommended = skill_selection.get("selected_skills", [])
        if recommended:
            skill_list = ", ".join(f'"{s}"' for s in recommended)
            out["skill_hint"] = (
                f"[Skill Recommendation] Your code matches patterns from: {skill_list}. "
                f"Next time, call `tool_search_and_load_skill({recommended[0]!r})` BEFORE writing Python "
                f"to show users the skill patterns you're following and ensure best practices."
            )
    
    # Separate figures (for frontend display) from content (for LLM)
    # This prevents large base64 images from being sent to LLM on subsequent calls
    figures = out.pop("figures", None)
    artifact = {"figures": figures} if figures else {}
    
    # Include figure references in content for LLM (without base64 images)
    # This allows the agent to reference figures in its response using [[fig:id|title]]
    if figures:
        figure_refs = []
        for fig in figures:
            fig_id = fig.get("id", "")
            title = fig.get("title", "Figure")
            reference = fig.get("reference", f"[[fig:{fig_id}|{title}]]")
            figure_refs.append({
                "id": fig_id,
                "title": title,
                "reference": reference,
            })
        out["generated_figures"] = figure_refs
    
    return out, artifact


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
# COMPOSITE TOOLS - Reduce round-trips by combining common multi-step workflows
# =============================================================================


@tool
def tool_stock_snapshot(query: str) -> dict:
    """One-call snapshot: search stock → resolve → latest prices + valuation + company info.

    Replaces the common 4-6 step preamble:
      search_stocks → get_current_datetime → get_dataset_status → get_daily_prices
      → get_daily_basic → get_stock_company

    This tool does ALL of the above in a single call.

    Args:
        query: Stock name, code, or keyword (e.g. '茅台', '300888', '600519.SH')

    Returns:
        {
          stock: {ts_code, name, industry, ...},
          now: {date, date_compact, current_trade_date},
          data_freshness: {conclusion, categories: [...]},
          latest_prices: [{trade_date, open, high, low, close, vol, pct_chg}, ...],  # last 10
          latest_valuation: [{trade_date, pe_ttm, pb, total_mv, circ_mv, turnover_rate}, ...],  # last 5
          company: {chairman, employees, main_business, ...} | null,
        }
    """
    result: dict = {}

    # 1) Resolve ts_code: try direct resolve first, then search
    stock_info = None
    ts_code = None

    # If query looks like a ts_code (e.g. 600519.SH), try resolve directly
    if "." in query and len(query) <= 12:
        try:
            resolved = resolve_symbol(query, store_dir=STORE_DIR)
            if resolved.get("ts_code"):
                ts_code = resolved["ts_code"]
                stock_info = resolved
        except Exception:
            pass

    if ts_code is None:
        search_result = search_stocks(query, offset=0, limit=5, store_dir=STORE_DIR)
        rows = search_result.get("rows") or []
        if rows:
            top = rows[0]
            ts_code = top.get("ts_code")
            stock_info = top
        else:
            return {"error": f"未找到匹配 '{query}' 的股票", "stock": None}

    result["stock"] = stock_info

    # 2) Current datetime
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz)
    result["now"] = {
        "date": now.strftime("%Y-%m-%d"),
        "date_compact": now.strftime("%Y%m%d"),
        "time": now.strftime("%H:%M"),
    }

    # 3) Data freshness (lightweight: skip full dataset_status, just check market category)
    try:
        ds = tool_get_dataset_status.invoke({})
        result["data_freshness"] = {
            "conclusion": ds.get("conclusion", ""),
            "categories": ds.get("categories", []),
        }
    except Exception:
        result["data_freshness"] = {"conclusion": "无法获取", "categories": []}

    # 4) Latest prices (last 10 trading days)
    try:
        prices = get_daily_prices(
            ts_code, start_date=None, end_date=None, offset=0, limit=10, store_dir=STORE_DIR
        )
        result["latest_prices"] = prices.get("rows", [])
    except Exception:
        result["latest_prices"] = []

    # 5) Latest valuation (last 5 rows)
    try:
        basic = get_daily_basic(
            ts_code, start_date=None, end_date=None, offset=0, limit=5, store_dir=STORE_DIR
        )
        result["latest_valuation"] = basic.get("rows", [])
    except Exception:
        result["latest_valuation"] = []

    # 6) Company profile
    try:
        company = get_stock_company(ts_code, store_dir=STORE_DIR)
        result["company"] = company if company.get("found") else None
    except Exception:
        result["company"] = None

    return result


@tool
def tool_smart_search(
    query: str,
    search_types: list[str] | None = None,
    limit: int = 10,
) -> dict:
    """Search across stocks, indices, and ETFs/funds in one call.

    Replaces the trial-and-error pattern of calling search_stocks → get_index_basic
    → get_fund_basic sequentially until a match is found.

    Args:
        query: Name, code, or keyword (e.g. '沪深300', '黄金ETF', '茅台')
        search_types: List of types to search. Default: all three.
            Options: 'stock', 'index', 'fund'
        limit: Max results per type (default 10)

    Returns:
        {
          stock_results: {rows, total_count} | null,
          index_results: {rows, total_count} | null,
          fund_results: {rows, total_count} | null,
          best_match: {type, ts_code, name} | null,
        }
    """
    import re as _re

    types = search_types or ["stock", "index", "fund"]
    cap = min(limit or 10, 50)
    result: dict = {}

    best_match = None

    # --- detect code-like queries so we also do ts_code exact lookups ---
    # Matches bare 6-digit codes (e.g. '513800') or codes with exchange
    # suffix already present (e.g. '513800.SH', '161128.sz').
    _bare_code_re = _re.compile(r'^(\d{6})$')
    _full_code_re = _re.compile(r'^(\d{6})\.(SH|SZ|BJ)$', _re.I)

    _bare_m = _bare_code_re.match(query.strip())
    _full_m = _full_code_re.match(query.strip())

    if _bare_m:
        # Try both exchanges for fund and index ts_code lookups
        _code_candidates = [f"{_bare_m.group(1)}.SH", f"{_bare_m.group(1)}.SZ"]
    elif _full_m:
        _code_candidates = [query.strip().upper()]
    else:
        _code_candidates = []

    def _merge_rows(existing: dict | None, extra_rows: list) -> dict:
        """Merge extra_rows into an existing result dict, deduplicating by ts_code."""
        if not extra_rows:
            return existing or {"rows": [], "total_count": 0, "showing": "0-0", "has_more": False}
        if not existing or not existing.get("rows"):
            merged_rows = extra_rows
        else:
            seen = {r.get("ts_code") for r in existing["rows"]}
            merged_rows = extra_rows + [r for r in existing["rows"] if r.get("ts_code") not in seen]
        total = len(merged_rows)
        return {
            "rows": merged_rows,
            "total_count": total,
            "showing": f"1-{total}",
            "has_more": False,
        }

    if "stock" in types:
        try:
            sr = search_stocks(query, offset=0, limit=cap, store_dir=STORE_DIR)
            result["stock_results"] = sr
            rows = sr.get("rows") or []
            if rows and best_match is None:
                best_match = {"type": "stock", "ts_code": rows[0].get("ts_code"), "name": rows[0].get("name")}
        except Exception:
            result["stock_results"] = None

    if "index" in types:
        try:
            ir = get_index_basic(
                name_contains=query, offset=0, limit=cap, store_dir=STORE_DIR
            )
            # Supplement with ts_code-based lookup for code queries
            code_rows: list = []
            for _candidate in _code_candidates:
                try:
                    _cr = get_index_basic(ts_code=_candidate, store_dir=STORE_DIR)
                    code_rows.extend(_cr.get("rows") or [])
                except Exception:
                    pass
            ir = _merge_rows(ir, code_rows)
            result["index_results"] = ir
            rows = ir.get("rows") or []
            if rows and best_match is None:
                best_match = {"type": "index", "ts_code": rows[0].get("ts_code"), "name": rows[0].get("name")}
        except Exception:
            result["index_results"] = None

    if "fund" in types:
        try:
            fr = get_fund_basic(
                name_contains=query, offset=0, limit=cap, store_dir=STORE_DIR
            )
            # Supplement with ts_code-based lookup for code queries
            code_rows = []
            for _candidate in _code_candidates:
                try:
                    _cr = get_fund_basic(ts_code=_candidate, store_dir=STORE_DIR)
                    code_rows.extend(_cr.get("rows") or [])
                except Exception:
                    pass
            fr = _merge_rows(fr, code_rows)
            result["fund_results"] = fr
            rows = fr.get("rows") or []
            if rows and best_match is None:
                best_match = {"type": "fund", "ts_code": rows[0].get("ts_code"), "name": rows[0].get("name")}
        except Exception:
            result["fund_results"] = None

    result["best_match"] = best_match
    return result


@tool
def tool_peer_comparison(
    ts_code: str,
    metrics: list[str] | None = None,
    limit: int = 10,
) -> dict:
    """Compare a stock with its industry peers on key valuation/size metrics.

    Auto-detects the stock's industry, fetches peer stocks, and returns a
    comparison table — all in one call.

    Args:
        ts_code: Stock ts_code (e.g. '600519.SH')
        metrics: Valuation columns to include. Default: ['pe_ttm', 'pb', 'total_mv']
            Options: pe_ttm, pb, ps_ttm, total_mv, circ_mv, turnover_rate
        limit: Max number of peers to return (default 10)

    Returns:
        {
          target: {ts_code, name, industry, ...metrics},
          peers: [{ts_code, name, industry, ...metrics}, ...],
          industry: str,
          peer_count: int,
        }
    """
    metrics = metrics or ["pe_ttm", "pb", "total_mv"]
    cap = min(limit or 10, 30)

    # 1) Get target stock basic info to find industry
    target_info = get_stock_basic_detail(ts_code, store_dir=STORE_DIR)
    if not target_info.get("found"):
        return {"error": f"未找到股票 {ts_code}", "target": None, "peers": []}

    target_data = target_info.get("data", {})
    industry = target_data.get("industry", "")
    if not industry:
        return {"error": f"无法获取 {ts_code} 行业信息", "target": target_data, "peers": []}

    # 2) Get universe of peers in the same industry
    universe = get_universe(industry=industry, offset=0, limit=cap + 5, store_dir=STORE_DIR)
    peer_codes = [r.get("ts_code") for r in (universe.get("rows") or []) if r.get("ts_code")]

    # Ensure target is in the list
    if ts_code not in peer_codes:
        peer_codes.insert(0, ts_code)

    # 3) Fetch latest valuation for each peer
    def _get_valuation(code: str) -> dict | None:
        try:
            basic = get_daily_basic(code, start_date=None, end_date=None, offset=0, limit=1, store_dir=STORE_DIR)
            rows = basic.get("rows") or []
            if rows:
                row = rows[0]
                entry = {"ts_code": code}
                # Find name from universe rows
                for u in (universe.get("rows") or []):
                    if u.get("ts_code") == code:
                        entry["name"] = u.get("name", "")
                        break
                for m in metrics:
                    entry[m] = row.get(m)
                return entry
        except Exception:
            pass
        return None

    target_val = None
    peers = []
    for code in peer_codes[:cap + 1]:
        val = _get_valuation(code)
        if val is None:
            continue
        if code == ts_code:
            val["name"] = val.get("name") or target_data.get("name", "")
            target_val = val
        else:
            peers.append(val)
        if len(peers) >= cap:
            break

    return {
        "target": target_val,
        "peers": peers,
        "industry": industry,
        "peer_count": len(peers),
    }


@tool
def tool_search_and_load_skill(query_or_skill_id: str) -> dict:
    """Search for a skill and load its full content in one call.

    Replaces the two-step pattern: tool_search_skills → tool_load_skill.

    If `query_or_skill_id` exactly matches a skill directory name, loads it directly.
    Otherwise, searches by keyword and loads the top result.

    Args:
        query_or_skill_id: Exact skill ID (e.g. 'rolling_indicators') or search query
            (e.g. '均线计算', 'backtest', 'MACD')

    Returns:
        {
          found: bool,
          skill_id: str,
          name: str,
          content: str,    # Full skill markdown body
          meta: {...},     # Frontmatter metadata
          alternatives: [{skill_id, name, description}, ...],  # Other matches
        }
    """
    # Try direct load first
    loaded = load_skill(query_or_skill_id)
    if loaded.get("found"):
        # Also provide alternatives
        alts = []
        try:
            hits = search_skills(query_or_skill_id, k=3)
            for s in hits:
                sid = s.path.parent.name
                if sid != query_or_skill_id:
                    alts.append({"skill_id": sid, "name": s.name, "description": s.description})
        except Exception:
            pass
        return {
            "found": True,
            "skill_id": query_or_skill_id,
            "name": loaded.get("name", query_or_skill_id),
            "content": loaded.get("content", ""),
            "meta": loaded.get("meta", {}),
            "alternatives": alts,
        }

    # Search by keyword
    hits = search_skills(query_or_skill_id, k=5)
    if not hits:
        return {
            "found": False,
            "skill_id": None,
            "name": None,
            "content": None,
            "meta": {},
            "alternatives": [],
            "message": f"未找到与 '{query_or_skill_id}' 相关的 skill",
        }

    # Load the top result
    top = hits[0]
    top_id = top.path.parent.name
    loaded = load_skill(top_id)

    alts = []
    for s in hits[1:]:
        sid = s.path.parent.name
        alts.append({"skill_id": sid, "name": s.name, "description": s.description})

    return {
        "found": loaded.get("found", False),
        "skill_id": top_id,
        "name": loaded.get("name", top.name),
        "content": loaded.get("content", ""),
        "meta": loaded.get("meta", {}),
        "alternatives": alts,
    }


@tool(response_format="content_and_artifact")
def tool_backtest_strategy(
    ts_codes: list[str],
    strategy: str,
    params: dict | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    fee_rate: float | None = None,
    compare_stocks: bool = False,
) -> tuple[dict, dict]:
    """Run a strategy backtest on one or more stocks — NO Python code needed!

    This replaces writing 50+ lines of backtest Python. Returns metrics, a
    comparison table, and an equity-curve chart automatically.

    ## Supported Strategies

    | strategy      | Description                          | Key params (defaults)                           |
    |---------------|--------------------------------------|-------------------------------------------------|
    | `dual_ma`     | Dual moving-average crossover        | fast=5, slow=20, ma_type="sma" (or "ema")      |
    | `bollinger`   | Bollinger-band mean-reversion        | period=20, num_std=2.0                          |
    | `macd`        | MACD histogram crossover             | fast=12, slow=26, signal=9                      |
    | `chandelier`  | Chandelier ATR trailing-stop exit    | atr_period=22, mult=3.0                         |
    | `buy_and_hold`| Simple buy-and-hold                  | (none)                                          |
    | `momentum`    | Momentum rotation (multi-asset)      | n_days=20, top_k=1, rebal_freq=20              |

    ## Usage Examples

    **Single stock, default params:**
    ```
    tool_backtest_strategy(ts_codes=["600519.SH"], strategy="dual_ma")
    ```

    **Custom params:**
    ```
    tool_backtest_strategy(ts_codes=["600519.SH"], strategy="dual_ma",
                           params={"fast": 10, "slow": 60, "ma_type": "ema"})
    ```

    **Compare same strategy across multiple stocks:**
    ```
    tool_backtest_strategy(ts_codes=["600519.SH", "000858.SZ", "000568.SZ"],
                           strategy="macd", compare_stocks=True)
    ```

    **Momentum rotation (multi-asset portfolio):**
    ```
    tool_backtest_strategy(ts_codes=["600519.SH", "601318.SH", "000858.SZ"],
                           strategy="momentum",
                           params={"n_days": 60, "top_k": 1, "rebal_freq": 20})
    ```

    Args:
        ts_codes: One or more stock/ETF ts_codes (e.g. ["600519.SH"])
        strategy: Strategy name (see table above)
        params: Override default strategy parameters
        start_date: Backtest start date YYYYMMDD (default: all available data)
        end_date: Backtest end date YYYYMMDD (default: latest data)
        fee_rate: Transaction fee per side (default: 0.03% single-asset, 0.1% momentum)
        compare_stocks: If True and multiple ts_codes, compare them side by side

    Returns:
        Tuple of (content, artifact):
        - content: {strategy, params, results: [{ts_code, name, metrics}], comparison_table, errors}
        - artifact: {figures: [{image, title, format}]} for frontend chart display
    """
    from agent.backtest import run_backtest
    from agent.figures import save_figure, format_figure_reference
    from agent.sandbox import get_thread_id

    result = run_backtest(
        ts_codes=ts_codes,
        strategy=strategy,
        params=params,
        start_date=start_date,
        end_date=end_date,
        fee_rate=fee_rate,
        generate_chart=True,
    )

    # Separate figures from content (same pattern as tool_execute_python)
    raw_figures = result.pop("figures", None) or []
    artifact_figures = []
    figure_refs = []

    tid = get_thread_id() or "default"
    for fig in raw_figures:
        img_b64 = fig.get("image", "")
        title = fig.get("title", "Backtest")
        fig_meta = save_figure(image_base64=img_b64, title=title, format="png", thread_id=tid)
        fig_id = fig_meta["id"]
        ref = format_figure_reference(fig_id, title)
        artifact_figures.append({
            "id": fig_id,
            "title": title,
            "format": "png",
            "image": img_b64,
            "reference": ref,
        })
        figure_refs.append({"id": fig_id, "title": title, "reference": ref})

    # Content for LLM (no base64 images)
    result["generated_figures"] = figure_refs
    artifact = {"figures": artifact_figures} if artifact_figures else {}

    return result, artifact


# =============================================================================
# EXPORT
# =============================================================================

ALL_TOOLS = [
    # Composite (high-level, reduces round-trips)
    tool_stock_snapshot,
    tool_smart_search,
    tool_peer_comparison,
    tool_search_and_load_skill,
    tool_backtest_strategy,
    # Discovery
    tool_list_industries,
    tool_get_universe,
    # Simple Data (focused lookups)
    tool_get_daily_prices,
    tool_get_daily_basic,
    tool_get_index_daily_prices,
    tool_get_etf_daily_prices,
    tool_get_fund_nav,
    tool_get_income,
    tool_get_balancesheet,
    tool_get_cashflow,
    tool_get_fina_indicator,
    tool_get_dividend,
    # Market Extras
    tool_get_moneyflow,
    tool_get_fx_daily,
    # Macro
    tool_get_lpr,
    tool_get_cpi,
    tool_get_cn_sf,
    tool_get_cn_m,
    # Calendar
    tool_get_trading_days,
    tool_is_trading_day,
    tool_get_prev_trade_date,
    tool_get_next_trade_date,
    # Python Execution (for complex analysis only)
    tool_execute_python,
]
