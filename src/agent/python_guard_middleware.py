"""Middleware guardrails for Python tool usage.

Goals:
1) Prevent calling tool_execute_python for simple lookups (prices/PE/company info).
2) Prevent "print-only" python calls that do no real analysis.

This is a pragmatic heuristic layer to keep the agent efficient and effective.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import ToolMessage
from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest


def _last_human_text(state: Any) -> str:
    try:
        if not isinstance(state, dict):
            return ""
        msgs = state.get("messages")
        if not isinstance(msgs, list):
            return ""
        for m in reversed(msgs):
            if m.__class__.__name__ == "HumanMessage":
                c = getattr(m, "content", "")
                return c if isinstance(c, str) else str(c)
        return ""
    except Exception:
        return ""


def _is_simple_lookup_query(q: str) -> bool:
    """Heuristic: query can be answered by non-python tools."""
    s = (q or "").strip()
    if not s:
        return False

    # Calculation / deep analysis hints: allow python
    calc_terms = [
        "计算",
        "回测",
        "策略",
        "指标",
        "均线",
        "MA",
        "RSI",
        "MACD",
        "波动",
        "相关",
        "回归",
        "对比",
        "比较",
        "预测",
        "因子",
        "分位",
        "统计",
        "收益率",
        "涨幅",
        "跌幅",
        "年化",
        "最大回撤",
    ]
    if any(t in s for t in calc_terms):
        return False

    # Simple lookup intents: prefer simple tools
    lookup_terms = [
        "最近",
        "最新",
        "股价",
        "收盘",
        "开盘",
        "最高",
        "最低",
        "成交量",
        "涨跌幅",
        "估值",
        "PE",
        "PB",
        "市值",
        "换手",
        "公司信息",
        "公司简介",
        "主营",
        "所属行业",
        "列出",
        "有哪些",
    ]
    return any(t in s for t in lookup_terms)


def _is_print_only_code(code: str) -> bool:
    return False
    # Disabled for now - the heuristic is too aggressive
    """Heuristic: code mostly prints / lists without computing anything meaningful."""


def _reject(tool_call_id: str | None, message: str) -> ToolMessage:
    return ToolMessage(content=message, tool_call_id=tool_call_id or "unknown")


# Rejection message for simple lookup queries
_SIMPLE_LOOKUP_REJECTION = (
    "拒绝执行：这个问题属于「简单查询/信息获取」，不需要使用 Python。\n"
    "请改用相应工具：\n"
    "- 股价+估值+公司信息：tool_stock_snapshot(query) — 一次获取全部\n"
    "- 股价详情：tool_get_daily_prices(ts_code, limit)\n"
    "- 估值详情：tool_get_daily_basic(ts_code, limit)\n"
    "- 行业/清单：tool_get_universe 或 tool_smart_search\n"
    "- 同行对比：tool_peer_comparison(ts_code)\n"
)

# Rejection message for print-only code
_PRINT_ONLY_REJECTION = (
    "拒绝执行：你准备运行的 Python 代码几乎只是在打印/罗列数据，没有进行必要的计算分析。\n"
    "请优先使用简单工具获取数据，或在 Python 中加入明确的计算（例如收益率/均线/波动率/区间涨跌/对比表）。"
)


class PythonGuardMiddleware(AgentMiddleware[Any, Any]):
    """Guardrails for Python tool usage."""

    tools = []

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]) -> Any:
        tool_call = request.tool_call or {}
        name = tool_call.get("name")
        if name != "tool_execute_python":
            return handler(request)

        args = tool_call.get("args") or {}
        code = (args.get("code") or "").strip()

        q = _last_human_text(getattr(request, "state", None))
        tool_call_id = tool_call.get("id")

        # 1) Never use python for simple lookups
        if _is_simple_lookup_query(q):
            return _reject(tool_call_id, _SIMPLE_LOOKUP_REJECTION)

        # 2) Block print-only / non-computational python calls
        if _is_print_only_code(code):
            return _reject(tool_call_id, _PRINT_ONLY_REJECTION)

        return handler(request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]
    ) -> Any:
        tool_call = request.tool_call or {}
        name = tool_call.get("name")
        if name != "tool_execute_python":
            return await handler(request)

        args = tool_call.get("args") or {}
        code = (args.get("code") or "").strip()

        q = _last_human_text(getattr(request, "state", None))
        tool_call_id = tool_call.get("id")

        if _is_simple_lookup_query(q):
            return _reject(tool_call_id, _SIMPLE_LOOKUP_REJECTION)

        if _is_print_only_code(code):
            return _reject(tool_call_id, _PRINT_ONLY_REJECTION)

        return await handler(request)

