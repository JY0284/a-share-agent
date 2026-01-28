"""Middleware guardrails for Python tool usage.

Goals:
1) Prevent calling tool_execute_python for simple lookups (prices/PE/company info).
2) Prevent "print-only" python calls that do no real analysis.
3) Enforce skills usage for any Python execution (skills_used must be non-empty).

This is a pragmatic heuristic layer to keep the agent efficient and sane.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.messages import AIMessage
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
    """Heuristic: code mostly prints / lists without computing anything meaningful."""
    lines = []
    for raw in (code or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)

    if not lines:
        return True

    # If it doesn't access the store at all, it's almost certainly pointless here.
    if "store." not in (code or ""):
        # Allow pure computation without store (rare)
        compute_markers = ["rolling(", "pct_change(", ".diff(", ".mean(", ".std(", "np.", "pd."]
        if not any(m in (code or "") for m in compute_markers):
            return True

    # If all non-import statements are print / display
    meaningful = 0
    for ln in lines:
        if ln.startswith("import ") or ln.startswith("from "):
            continue
        if ln.startswith("print("):
            continue
        # allow assignments, but count as meaningful only if it includes computation markers
        compute_markers = [
            "rolling(",
            "pct_change(",
            ".diff(",
            ".mean(",
            ".std(",
            "corr(",
            "groupby(",
            "resample(",
            "np.",
        ]
        if any(m in ln for m in compute_markers):
            meaningful += 1
        else:
            # plain data fetch/formatting is not meaningful enough
            pass

    # If no compute markers at all, treat as print-only / lookup code
    return meaningful == 0


def _reject(tool_call_id: str | None, message: str) -> ToolMessage:
    return ToolMessage(content=message, tool_call_id=tool_call_id or "unknown")


def _loaded_skills_in_state(state: Any) -> dict[str, bool]:
    """Return {skill_id: found_bool} for skills that were loaded via tool_load_skill."""
    out: dict[str, bool] = {}
    try:
        if not isinstance(state, dict):
            return out
        msgs = state.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return out

        # Map tool_call_id -> skill_id for tool_load_skill calls
        load_calls: dict[str, str] = {}
        for m in msgs:
            if not isinstance(m, AIMessage):
                continue
            for tc in getattr(m, "tool_calls", []) or []:
                if not isinstance(tc, dict):
                    continue
                if tc.get("name") != "tool_load_skill":
                    continue
                args = tc.get("args") or {}
                skill_id = args.get("skill_id")
                call_id = tc.get("id")
                if skill_id and call_id:
                    load_calls[str(call_id)] = str(skill_id)

        if not load_calls:
            return out

        # Find corresponding ToolMessages and parse found field
        for m in msgs:
            if not isinstance(m, ToolMessage):
                continue
            call_id = getattr(m, "tool_call_id", None)
            if not call_id:
                continue
            skill_id = load_calls.get(str(call_id))
            if not skill_id:
                continue
            found = False
            try:
                payload = json.loads(m.content) if isinstance(m.content, str) else {}
                found = bool(payload.get("found"))
            except Exception:
                # fallback heuristic
                if isinstance(m.content, str) and '"found": true' in m.content.lower():
                    found = True
            out[skill_id] = found
        return out
    except Exception:
        return out


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
        skills_used = args.get("skills_used") or []

        q = _last_human_text(getattr(request, "state", None))
        tool_call_id = tool_call.get("id")

        # 1) Never use python for simple lookups
        if _is_simple_lookup_query(q):
            return _reject(
                tool_call_id,
                "拒绝执行：这个问题属于“简单查询/信息获取”，不需要使用 Python。\n"
                "请改用相应工具：\n"
                "- 股价：tool_get_daily_prices(ts_code, limit)\n"
                "- 估值：tool_get_daily_basic(ts_code, limit)\n"
                "- 公司信息：tool_get_stock_basic_detail / tool_get_stock_company\n"
                "- 行业/清单：tool_get_universe 或 tool_search_stocks\n",
            )

        # 2) Enforce skills_used for python
        if not isinstance(skills_used, list) or len(skills_used) == 0:
            return _reject(
                tool_call_id,
                "拒绝执行：使用 Python 进行分析前，必须先加载 1-3 个 skills，并在调用 tool_execute_python 时传入 skills_used。\n"
                "建议流程：tool_search_skills(query=你的分析目标, limit=3) → tool_load_skill(...) → tool_execute_python(..., skills_used=[...])\n"
                "如果不确定选哪个技能，至少加载 robust_df_checks + trading_day_windows。",
            )

        # 2.5) Enforce that declared skills were actually loaded via tool_load_skill
        loaded = _loaded_skills_in_state(getattr(request, "state", None))
        missing = [s for s in skills_used if s not in loaded]
        not_found = [s for s in skills_used if s in loaded and not loaded[s]]
        if missing or not_found:
            return _reject(
                tool_call_id,
                "拒绝执行：你在 skills_used 里声明了技能，但本轮对话里还没有成功加载这些技能内容。\n"
                f"- 未加载: {missing}\n"
                f"- 加载失败(found=false): {not_found}\n"
                "请先调用 tool_load_skill(skill_id=...) 加载技能内容（建议 1-3 个），再执行 Python。",
            )

        # 3) Block print-only / non-computational python calls
        if _is_print_only_code(code):
            return _reject(
                tool_call_id,
                "拒绝执行：你准备运行的 Python 代码几乎只是在打印/罗列数据，没有进行必要的计算分析。\n"
                "请优先使用简单工具获取数据，或在 Python 中加入明确的计算（例如收益率/均线/波动率/区间涨跌/对比表）。",
            )

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
        skills_used = args.get("skills_used") or []

        q = _last_human_text(getattr(request, "state", None))
        tool_call_id = tool_call.get("id")

        if _is_simple_lookup_query(q):
            return _reject(
                tool_call_id,
                "拒绝执行：这个问题属于“简单查询/信息获取”，不需要使用 Python。\n"
                "请改用相应工具：\n"
                "- 股价：tool_get_daily_prices(ts_code, limit)\n"
                "- 估值：tool_get_daily_basic(ts_code, limit)\n"
                "- 公司信息：tool_get_stock_basic_detail / tool_get_stock_company\n"
                "- 行业/清单：tool_get_universe 或 tool_search_stocks\n",
            )

        if not isinstance(skills_used, list) or len(skills_used) == 0:
            return _reject(
                tool_call_id,
                "拒绝执行：使用 Python 进行分析前，必须先加载 1-3 个 skills，并在调用 tool_execute_python 时传入 skills_used。\n"
                "建议流程：tool_search_skills(query=你的分析目标, limit=3) → tool_load_skill(...) → tool_execute_python(..., skills_used=[...])\n"
                "如果不确定选哪个技能，至少加载 robust_df_checks + trading_day_windows。",
            )

        loaded = _loaded_skills_in_state(getattr(request, "state", None))
        missing = [s for s in skills_used if s not in loaded]
        not_found = [s for s in skills_used if s in loaded and not loaded[s]]
        if missing or not_found:
            return _reject(
                tool_call_id,
                "拒绝执行：你在 skills_used 里声明了技能，但本轮对话里还没有成功加载这些技能内容。\n"
                f"- 未加载: {missing}\n"
                f"- 加载失败(found=false): {not_found}\n"
                "请先调用 tool_load_skill(skill_id=...) 加载技能内容（建议 1-3 个），再执行 Python。",
            )

        if _is_print_only_code(code):
            return _reject(
                tool_call_id,
                "拒绝执行：你准备运行的 Python 代码几乎只是在打印/罗列数据，没有进行必要的计算分析。\n"
                "请优先使用简单工具获取数据，或在 Python 中加入明确的计算（例如收益率/均线/波动率/区间涨跌/对比表）。",
            )

        return await handler(request)

