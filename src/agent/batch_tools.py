"""Batch data-fetching tools for portfolio-level operations.

These tools fetch data for MULTIPLE assets in a single agent tool-call,
eliminating the 10+ sequential round-trips seen in traces.

Design:
- tool_batch_quotes: fetch latest price for a list of ts_codes (stocks, ETFs, indices)
- tool_portfolio_live_snapshot: fetch prices for all user holdings + key indices
- tool_market_overview: fetch major indices + macro signals in one call
- tool_compare_stocks: side-by-side comparison of 2-5 stocks in one call

All functions are synchronous (matching existing tool conventions) and use
the same store_dir / agent_tools layer as the rest of the codebase.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from stock_data.agent_tools import (
    get_daily_prices,
    get_daily_basic,
    get_index_daily_prices,
    get_etf_daily_prices,
    get_fina_indicator,
    get_stock_basic_detail,
    get_stock_company,
    search_stocks,
    get_lpr,
    get_cpi,
    get_cn_m,
    get_fx_daily,
)

from agent.user_profile import get_or_create_profile

if TYPE_CHECKING:
    from agent.user_profile import Holding

STORE_DIR = os.environ.get("STOCK_DATA_STORE_DIR", "../stock_data/store")

# Major indices to always include in market overview
KEY_INDICES = [
    ("000001.SH", "上证指数"),
    ("399001.SZ", "深证成指"),
    ("399006.SZ", "创业板指"),
    ("000300.SH", "沪深300"),
    ("000905.SH", "中证500"),
]

# Backward-compat alias for any code still importing the private name
_KEY_INDICES = KEY_INDICES


def fetch_latest_price(ts_code: str, asset_type: str = "stock") -> dict:
    """Fetch latest 1-day price for a single ts_code. Returns a summary dict."""
    try:
        if asset_type in ("index", "index_etf_underlying"):
            rows = get_index_daily_prices(
                ts_code, start_date=None, end_date=None, offset=0, limit=1, store_dir=STORE_DIR
            ).get("rows", [])
        elif asset_type in ("etf", "qdii", "index_etf", "bond_fund"):
            rows = get_etf_daily_prices(
                ts_code, start_date=None, end_date=None, offset=0, limit=1, store_dir=STORE_DIR
            ).get("rows", [])
        else:
            rows = get_daily_prices(
                ts_code, start_date=None, end_date=None, offset=0, limit=1, store_dir=STORE_DIR
            ).get("rows", [])
        if rows:
            return {"ts_code": ts_code, "found": True, **rows[0]}
        return {"ts_code": ts_code, "found": False}
    except Exception as e:
        return {"ts_code": ts_code, "found": False, "error": str(e)}


# Backward-compat alias
_fetch_latest_price = fetch_latest_price


def _detect_asset_type(code: str) -> str:
    """Best-effort asset type detection from ts_code prefix/suffix."""
    numeric = code.split(".")[0]
    suffix = code.split(".")[-1] if "." in code else ""

    # Shanghai/Shenzhen indices
    if numeric.startswith("000") and suffix == "SH":
        return "index"
    if numeric.startswith("399") and suffix == "SZ":
        return "index"

    # ETFs — cover all common prefixes:
    #   51xxxx.SH  (stock/bond/commodity ETFs on SSE)
    #   15xxxx.SZ  (stock/bond/commodity ETFs on SZSE)
    #   56xxxx.SH  (cross-border ETFs)
    #   16xxxx.SZ  (LOF/QDII funds)
    if len(numeric) == 6 and numeric[:2] in ("51", "15", "56", "16"):
        return "etf"

    return "stock"


@tool
def tool_batch_quotes(
    ts_codes: list[str],
    asset_types: list[str] | None = None,
) -> dict:
    """Fetch latest price for multiple assets in one call.

    Use this to get current prices for a list of stocks/ETFs/indices instead
    of calling tool_get_daily_prices / tool_get_etf_daily_prices one by one.

    Args:
        ts_codes: List of ts_codes, e.g. ["600519.SH", "513800.SH", "000300.SH"]
        asset_types: Parallel list of asset types for each ts_code.
            Options per element: "stock" | "etf" | "qdii" | "index" | "index_etf"
            If omitted, auto-detects based on ts_code patterns.

    Returns:
        {quotes: [{ts_code, found, trade_date, close, pct_chg, ...}], count: N}
    """
    types = list(asset_types or [])

    # Auto-detect asset types for any missing entries
    while len(types) < len(ts_codes):
        types.append(_detect_asset_type(ts_codes[len(types)]))

    # Parallel fetch
    quotes: list[dict] = [None] * len(ts_codes)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=min(len(ts_codes), 8) or 1) as pool:
        futures = {
            pool.submit(fetch_latest_price, code, atype): idx
            for idx, (code, atype) in enumerate(zip(ts_codes, types))
        }
        for future in as_completed(futures):
            quotes[futures[future]] = future.result()

    return {"quotes": quotes, "count": len(quotes)}


@tool
def tool_portfolio_live_snapshot(*, config: RunnableConfig) -> dict:
    """Fetch live prices for ALL user holdings plus key market indices.

    This is the ONE-CALL tool for "向我汇报" / portfolio report:
    1. Loads the user's structured portfolio
    2. Fetches latest prices for every held asset
    3. Fetches key index levels (上证/深证/创业板/沪深300/中证500)
    4. Computes updated P&L

    Returns:
        {
          has_portfolio: bool,
          holdings_live: [{name, ts_code, old_price, live_price, live_pnl_pct, ...}],
          indices: [{name, ts_code, close, pct_chg}],
          portfolio_summary: {total_value_est, total_pnl_est, ...},
        }
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id") or "dev_user"
    profile = get_or_create_profile(user_id)

    if not profile.holdings:
        return {
            "has_portfolio": False,
            "message": "No portfolio data. Ask user to share holdings first.",
        }

    # Collect resolvable holdings and pre-fill unresolved ones
    holdings_live: list[dict | None] = []
    resolvable: list[tuple[int, Holding]] = []
    for h in profile.holdings:
        if not h.ts_code:
            holdings_live.append({
                "name": h.name,
                "ts_code": "",
                "status": "unresolved",
                "shares": h.shares,
                "cost_price": h.cost_price,
                "last_known_price": h.current_price,
                "market_value": h.market_value,
            })
        else:
            resolvable.append((len(holdings_live), h))
            holdings_live.append(None)  # placeholder

    # Parallel-fetch all resolvable holdings + all key indices
    with ThreadPoolExecutor(max_workers=min(len(resolvable) + len(KEY_INDICES), 10) or 1) as pool:
        # Submit holding fetches
        h_futures = {
            pool.submit(fetch_latest_price, h.ts_code, h.asset_type): (idx, h)
            for idx, h in resolvable
        }
        # Submit index fetches
        idx_futures = {
            pool.submit(fetch_latest_price, idx_code, "index"): (idx_code, idx_name)
            for idx_code, idx_name in KEY_INDICES
        }

        # Collect holding results
        for future in as_completed(h_futures):
            idx, h = h_futures[future]
            quote = future.result()
            live_price = quote.get("close", h.current_price)
            live_date = quote.get("trade_date", "?")
            live_pct = quote.get("pct_chg", 0)

            if h.cost_price > 0 and live_price > 0:
                cost_pnl_pct = (live_price - h.cost_price) / h.cost_price
            else:
                cost_pnl_pct = h.pnl_pct

            live_value = h.shares * live_price if h.shares and live_price else h.market_value

            holdings_live[idx] = {
                "name": h.name,
                "ts_code": h.ts_code,
                "asset_type": h.asset_type,
                "shares": h.shares,
                "cost_price": h.cost_price,
                "last_known_price": h.current_price,
                "live_price": live_price,
                "live_date": live_date,
                "day_chg_pct": live_pct,
                "cost_pnl_pct": round(cost_pnl_pct, 5),
                "live_value": round(live_value, 2),
                "tags": h.tags,
            }

        # Collect index results
        indices = []
        for future in as_completed(idx_futures):
            idx_code, idx_name = idx_futures[future]
            quote = future.result()
            if quote.get("found"):
                indices.append({
                    "name": idx_name,
                    "ts_code": idx_code,
                    "close": quote.get("close"),
                    "pct_chg": quote.get("pct_chg"),
                    "trade_date": quote.get("trade_date"),
                })

    # Summary — filter out any None placeholders (shouldn't happen, but defensive)
    valid_holdings = [h for h in holdings_live if h is not None]
    total_live_value = sum(h.get("live_value", 0) for h in valid_holdings)
    total_cost = sum(
        (h.get("shares", 0) * h.get("cost_price", 0)) for h in valid_holdings
        if h.get("cost_price", 0) > 0
    )
    total_pnl = total_live_value - total_cost if total_cost > 0 else 0

    return {
        "has_portfolio": True,
        "holdings_live": valid_holdings,
        "indices": indices,
        "portfolio_summary": {
            "recorded_total_assets": profile.total_assets,
            "recorded_cash": profile.cash,
            "live_market_value_est": round(total_live_value, 2),
            "total_pnl_est": round(total_pnl, 2),
            "holding_count": len(holdings_live),
        },
        "preferences": profile.preferences.model_dump(),
        "updated_at": profile.updated_at,
    }


@tool
def tool_market_overview() -> dict:
    """Get a broad market overview: major indices + key macro indicators.

    Use this for general market reports or when the user asks about market conditions.
    Fetches in batch: 5 key indices, latest FX (USDCNH), latest LPR.

    Returns:
        {indices: [...], fx: {...}, macro: {lpr, cpi, m2}}
    """
    # Indices
    indices = []
    for idx_code, idx_name in KEY_INDICES:
        quote = fetch_latest_price(idx_code, "index")
        if quote.get("found"):
            indices.append({
                "name": idx_name,
                "ts_code": idx_code,
                "close": quote.get("close"),
                "pct_chg": quote.get("pct_chg"),
                "trade_date": quote.get("trade_date"),
            })

    # FX — USD/CNH
    fx = {}
    try:
        fx_data = get_fx_daily("USDCNH", start_date=None, end_date=None, offset=0, limit=1, store_dir=STORE_DIR)
        fx_rows = fx_data.get("rows", [])
        if fx_rows:
            fx = {"pair": "USDCNH", **fx_rows[0]}
    except Exception:
        pass

    # Macro
    macro = {}
    try:
        lpr_data = get_lpr(start_date=None, end_date=None, offset=0, limit=1, store_dir=STORE_DIR)
        lpr_rows = lpr_data.get("rows", [])
        if lpr_rows:
            macro["lpr_latest"] = lpr_rows[0]
    except Exception:
        pass

    try:
        cpi_data = get_cpi(month=None, offset=0, limit=1, store_dir=STORE_DIR)
        cpi_rows = cpi_data.get("rows", [])
        if cpi_rows:
            macro["cpi_latest"] = cpi_rows[0]
    except Exception:
        pass

    try:
        m_data = get_cn_m(month=None, offset=0, limit=1, store_dir=STORE_DIR)
        m_rows = m_data.get("rows", [])
        if m_rows:
            macro["money_supply_latest"] = m_rows[0]
    except Exception:
        pass

    return {
        "indices": indices,
        "fx": fx,
        "macro": macro,
    }


# ---------------------------------------------------------------------------
# tool_compare_stocks — side-by-side multi-stock comparison in one call
# ---------------------------------------------------------------------------

def _resolve_and_fetch(query: str) -> dict:
    """Resolve a stock name/code and fetch price + valuation + financials.

    Returns a single dict with all comparison-relevant fields, or an error
    marker if the stock cannot be resolved.
    """
    # 1. Resolve ts_code
    ts_code = None
    stock_info: dict = {}

    # If query looks like a ts_code already (e.g. "600519.SH"), use directly
    if "." in query and len(query) <= 12:
        ts_code = query.strip().upper()
        try:
            detail = get_stock_basic_detail(ts_code, store_dir=STORE_DIR)
            if detail.get("found"):
                stock_info = detail
        except Exception:
            pass

    if ts_code is None or not stock_info:
        try:
            sr = search_stocks(query, offset=0, limit=3, store_dir=STORE_DIR)
            rows = sr.get("rows") or []
            if rows:
                ts_code = rows[0].get("ts_code")
                stock_info = rows[0]
            else:
                return {"query": query, "resolved": False, "error": f"未找到匹配 '{query}' 的股票"}
        except Exception as e:
            return {"query": query, "resolved": False, "error": str(e)}

    result: dict = {
        "query": query,
        "resolved": True,
        "ts_code": ts_code,
        "name": stock_info.get("name", query),
        "industry": stock_info.get("industry", ""),
    }

    # 2. Latest price (1 row)
    try:
        px = get_daily_prices(ts_code, start_date=None, end_date=None, offset=0, limit=1, store_dir=STORE_DIR)
        rows = px.get("rows", [])
        if rows:
            result.update({
                "close": rows[0].get("close"),
                "pct_chg": rows[0].get("pct_chg"),
                "vol": rows[0].get("vol"),
                "trade_date": rows[0].get("trade_date"),
            })
    except Exception:
        pass

    # 3. Latest valuation (1 row: PE, PB, market cap)
    try:
        vl = get_daily_basic(ts_code, start_date=None, end_date=None, offset=0, limit=1, store_dir=STORE_DIR)
        rows = vl.get("rows", [])
        if rows:
            v = rows[0]
            result.update({
                "pe_ttm": v.get("pe_ttm"),
                "pb": v.get("pb"),
                "total_mv": v.get("total_mv"),
                "circ_mv": v.get("circ_mv"),
                "turnover_rate": v.get("turnover_rate"),
            })
    except Exception:
        pass

    # 4. Latest financial indicator (1 row: ROE, growth, etc.)
    try:
        fi = get_fina_indicator(ts_code, start_period=None, end_period=None, offset=0, limit=1, store_dir=STORE_DIR)
        rows = fi.get("rows", [])
        if rows:
            f = rows[0]
            result.update({
                "roe": f.get("roe"),
                "roe_dt": f.get("roe_dt"),
                "gross_profit_margin": f.get("grossprofit_margin"),
                "netprofit_margin": f.get("netprofit_margin"),
                "revenue_yoy": f.get("or_yoy"),  # operating revenue YoY
                "profit_yoy": f.get("netprofit_yoy"),
                "report_period": f.get("end_date"),
            })
    except Exception:
        pass

    # 5. Company info (main business, employees)
    try:
        co = get_stock_company(ts_code, store_dir=STORE_DIR)
        if co.get("found"):
            result.update({
                "main_business": co.get("main_business", ""),
                "employees": co.get("employees"),
            })
    except Exception:
        pass

    return result


@tool
def tool_compare_stocks(
    queries: list[str],
    metrics: list[str] | None = None,
) -> dict:
    """Compare 2-5 stocks side-by-side in ONE call.

    Resolves names, fetches prices + valuation + key financial indicators
    for all stocks IN PARALLEL. Use this for "对比X和Y", "compare X vs Y",
    or any multi-stock comparison query.

    Args:
        queries: List of stock names or ts_codes to compare (2-5 items).
            Examples: ["茅台", "五粮液"], ["600519.SH", "000858.SZ"]
        metrics: Optional list of metrics to highlight in the comparison.
            Default: all available (pe_ttm, pb, total_mv, roe, pct_chg, etc.)

    Returns:
        {
          stocks: [{name, ts_code, close, pct_chg, pe_ttm, pb, total_mv, roe, ...}],
          comparison_summary: str,  # formatted comparison table
          count: int,
        }
    """
    if len(queries) < 2:
        return {"error": "至少需要2只股票进行对比", "stocks": [], "count": 0}
    if len(queries) > 5:
        queries = queries[:5]  # cap at 5 to keep response manageable

    # Parallel resolve + fetch for all stocks
    stocks: list[dict] = [None] * len(queries)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as pool:
        futures = {
            pool.submit(_resolve_and_fetch, q): idx
            for idx, q in enumerate(queries)
        }
        for future in as_completed(futures):
            stocks[futures[future]] = future.result()

    # Build comparison summary text
    resolved = [s for s in stocks if s and s.get("resolved")]
    if not resolved:
        return {"error": "未能解析任何股票", "stocks": stocks, "count": 0}

    # Determine which metrics to show
    default_metrics = [
        "close", "pct_chg", "pe_ttm", "pb", "total_mv",
        "roe", "gross_profit_margin", "revenue_yoy", "profit_yoy",
    ]
    show_metrics = metrics or default_metrics

    # Build a simple comparison table as text
    header_labels = {
        "close": "最新价", "pct_chg": "涨跌幅%", "pe_ttm": "PE(TTM)",
        "pb": "PB", "total_mv": "总市值(万)", "circ_mv": "流通市值(万)",
        "turnover_rate": "换手率%", "roe": "ROE%", "roe_dt": "ROE(扣非)%",
        "gross_profit_margin": "毛利率%", "netprofit_margin": "净利率%",
        "revenue_yoy": "营收同比%", "profit_yoy": "净利同比%",
        "vol": "成交量", "trade_date": "数据日期",
    }

    lines = []
    header = "指标 | " + " | ".join(s.get("name", s.get("query", "?")) for s in resolved)
    lines.append(header)
    lines.append("-" * len(header))

    lines.append("代码 | " + " | ".join(s.get("ts_code", "?") for s in resolved))
    lines.append("行业 | " + " | ".join(s.get("industry", "?") for s in resolved))

    for m in show_metrics:
        label = header_labels.get(m, m)
        vals = []
        for s in resolved:
            v = s.get(m)
            if v is None:
                vals.append("-")
            elif isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append(f"{label} | " + " | ".join(vals))

    return {
        "stocks": stocks,
        "comparison_summary": "\n".join(lines),
        "count": len(resolved),
    }


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

BATCH_TOOLS = [
    tool_batch_quotes,
    tool_portfolio_live_snapshot,
    tool_market_overview,
    tool_compare_stocks,
]
