"""Skill system (agent-owned) for reusable analysis patterns.

Skills live at:
  a-share-agent/skills/<skill-name>/experience.md

Each experience.md contains:
- YAML frontmatter between --- markers (optional but recommended)
- Markdown body with guidance and examples

The system supports:
- Manual search+load via tool_search_and_load_skill (for user transparency)
- Auto-injection: smart_select_skills() picks best skills based on code + query
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Skill:
    name: str
    description: str | None
    tags: list[str]
    path: Path


# Cache for skill content (avoid repeated file reads)
_skill_content_cache: dict[str, dict[str, Any]] = {}


def _default_skills_dir() -> Path:
    # This file is: a-share-agent/src/agent/skills.py
    # parents[0]=agent, [1]=src, [2]=a-share-agent
    return Path(__file__).resolve().parents[2] / "skills"


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse simple YAML-ish frontmatter.

    We intentionally implement a minimal parser to avoid extra deps.
    Supported patterns:
      ---
      name: foo
      description: bar
      tags: [a, b]
      ---
      markdown body...
    """
    s = text.lstrip("\ufeff")
    if not s.startswith("---"):
        return {}, text

    lines = s.splitlines()
    if len(lines) < 3:
        return {}, text

    # Find closing '---'
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        return {}, text

    header_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1 :]).lstrip()

    meta: dict[str, Any] = {}
    for raw in header_lines:
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip()
        val = v.strip()

        # Very small conveniences
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            if inner == "":
                meta[key] = []
            else:
                meta[key] = [p.strip().strip("'\"") for p in inner.split(",")]
        else:
            meta[key] = val.strip("'\"")

    return meta, body


def list_skills(skills_dir: Path | None = None) -> list[Skill]:
    root = skills_dir or _default_skills_dir()
    if not root.exists():
        return []

    skills: list[Skill] = []
    for p in sorted(root.glob("*/experience.md")):
        txt = p.read_text(encoding="utf-8")
        meta, _body = _parse_frontmatter(txt)
        name = str(meta.get("name") or p.parent.name)
        desc = meta.get("description")
        tags_val = meta.get("tags") or []
        tags: list[str]
        if isinstance(tags_val, list):
            tags = [str(t) for t in tags_val if str(t).strip()]
        else:
            tags = [str(tags_val)]
        skills.append(Skill(name=name, description=desc, tags=tags, path=p))
    return skills


# Code pattern detection for auto-skill selection
_CODE_PATTERNS = {
    # Pattern -> skill candidates (ordered by priority)
    r"\.rolling\(|\.ewm\(|ma\d+|ema": ["rolling_indicators", "backtest_ma_crossover"],
    r"macd|ema_fast|ema_slow": ["rolling_indicators", "backtest_macd"],
    r"rsi|overbought|oversold": ["rolling_indicators"],
    r"atr|true_range|chandelier": ["rolling_indicators", "backtest_chandelier_exit"],
    r"bollinger|bb_upper|bb_lower|bb_mid": ["rolling_indicators", "backtest_bollinger"],
    r"sm\.|OLS|statsmodels|add_constant|\.fit\(\)": ["statistical_analysis"],
    r"alpha|beta|r_squared|regression|回归": ["statistical_analysis"],
    r"adfuller|coint|cointegration|协整|平稳": ["statistical_analysis"],
    r"arch_model|garch|volatility.*forecast|波动率.*预测": ["time_series_forecast"],
    r"ARIMA|arima|forecast|预测": ["time_series_forecast"],
    r"backtest|回测|strategy|策略|signal.*entry|signal.*exit": ["backtest_ma_crossover", "backtest_macd", "backtest_bollinger"],
    r"sharpe|cagr|max.*drawdown|mdd|equity.*curve": ["backtest_ma_crossover", "risk_metrics"],
    r"pct_change|收益率|returns|涨跌幅": ["adj_prices_and_returns"],
    r"qfq|hfq|复权|daily_adj": ["adj_prices_and_returns"],
    r"index_daily|指数|000300|399006": ["index_data", "index_returns_and_compare"],
    r"etf|fund_nav|nav|premium|折溢价": ["etf_data", "etf_nav_and_premium"],
    r"income|balancesheet|cashflow|fina_indicator|财务|利润|资产负债": ["finance_statements", "finance_statements_metrics"],
    r"pe_ttm|pb|ps_ttm|valuation|估值": ["valuation_units", "merge_prices_and_valuation"],
    r"corr|correlation|相关|compare|对比|比较": ["multi_stock_compare"],
    r"for.*ts_code|stocks\s*=\s*\[|多只|批量": ["parallel_multi_stock"],
    r"momentum|动量|breakout|突破|新高": ["momentum_breakout", "backtest_momentum_rotation"],
    r"trading_day|交易日|prev_trade|next_trade": ["trading_day_windows"],
    r"\.empty|KeyError|NameError|columns": ["robust_df_checks"],
}


def _score_skill(query: str, skill: Skill, code: str = "") -> int:
    """Score a skill based on query text and code patterns.
    
    Higher score = more relevant skill.
    """
    q = (query or "").strip()
    c = (code or "").strip()
    
    if not q and not c:
        return 0

    score = 0
    skill_id = skill.path.parent.name
    hay = " ".join(
        [
            skill.name,
            skill.description or "",
            " ".join(skill.tags),
            skill_id,
        ]
    ).lower()

    # 1) Code pattern matching (highest priority for auto-selection)
    if c:
        for pattern, skill_ids in _CODE_PATTERNS.items():
            if re.search(pattern, c, re.IGNORECASE):
                if skill_id in skill_ids:
                    # Higher boost for first match (primary skill)
                    idx = skill_ids.index(skill_id)
                    score += 20 - idx * 3  # 20 for primary, 17 for secondary, etc.

    # 2) Strong boosts for direct substring matches (Chinese-friendly)
    for needle in [skill.name, skill_id]:
        if needle and needle.lower() in q.lower():
            score += 15

    # 3) Token overlap (basic)
    q_tokens = [t.lower() for t in re.split(r'[\s,，、]+', q) if t]
    for t in q_tokens:
        if t in hay:
            score += 3

    # 4) Hand-tuned keyword boosts for common quant tasks
    boosts = {
        # backtest / strategy
        "回测": ["backtest", "回测", "equity", "drawdown", "sharpe", "cagr", "strategy", "策略"],
        "策略": ["strategy", "策略", "signal", "signals", "entries", "exits"],
        "双均线": ["ma", "均线", "crossover", "cross", "golden", "death"],
        "金叉": ["golden", "cross", "crossover", "金叉", "均线"],
        "死叉": ["death", "cross", "crossover", "死叉", "均线"],
        "均线": ["ma", "ma5", "ma20", "ma60", "rolling", "均线"],
        "MA": ["ma", "rolling"],
        "RSI": ["rsi"],
        "波动": ["std", "volatility", "波动"],
        "波动率": ["vol", "volatility", "波动率", "risk", "garch"],
        "风险": ["risk", "drawdown", "volatility", "风险", "回撤"],
        "回撤": ["drawdown", "mdd", "回撤"],
        "收益": ["returns", "ret", "pct_change", "收益", "涨跌幅", "复权"],
        "收益率": ["returns", "ret", "pct_change", "收益率"],
        "涨跌幅": ["pct_chg", "pct_change", "涨跌幅", "收益"],
        "动量": ["momentum", "mom", "动量", "breakout", "新高"],
        "突破": ["breakout", "donchian", "突破", "新高"],
        "新高": ["high52", "52w", "high", "新高", "breakout"],
        "复权": ["adj", "qfq", "hfq", "复权", "returns"],
        "相关": ["corr", "correlation", "相关", "compare"],
        "相关性": ["corr", "correlation", "相关性"],
        "对比": ["compare", "multi", "对比", "比较"],
        "比较": ["compare", "multi", "比较"],
        "合并": ["merge", "join", "合并"],
        "join": ["join", "merge"],
        "merge": ["merge", "join"],
        "市值": ["mv", "市值", "total_mv", "circ_mv", "亿元"],
        "换手": ["turnover", "换手"],
        "最近": ["sort", "descending", "tail", "head", "最近"],
        "trade_date": ["trade_date", "日期"],
        "日期": ["trade_date", "日期", "to_datetime"],
        # Statistical analysis
        "回归": ["regression", "OLS", "alpha", "beta", "回归", "stats"],
        "alpha": ["alpha", "regression", "capm"],
        "beta": ["beta", "regression", "capm"],
        "协整": ["coint", "cointegration", "协整", "配对"],
        "配对": ["pair", "coint", "cointegration", "配对"],
        "平稳": ["stationary", "adf", "平稳", "unit_root"],
        "GARCH": ["garch", "arch", "volatility", "波动率"],
        "预测": ["forecast", "arima", "预测", "garch"],
        "时间序列": ["time_series", "arima", "forecast", "时间序列"],
    }
    for k, keys in boosts.items():
        if k.lower() in q.lower():
            for kk in keys:
                if kk.lower() in hay:
                    score += 2

    return score


def search_skills(query: str, *, k: int = 5, code: str = "", skills_dir: Path | None = None) -> list[Skill]:
    """Search skills by query text and optionally by code patterns."""
    skills = list_skills(skills_dir)
    scored = [(s, _score_skill(query, s, code=code)) for s in skills]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, sc in scored[: max(0, int(k))] if sc > 0]


def load_skill(skill_name: str, *, skills_dir: Path | None = None, use_cache: bool = True) -> dict[str, Any]:
    """Load full skill content. Uses cache by default for performance."""
    global _skill_content_cache
    
    if use_cache and skill_name in _skill_content_cache:
        return _skill_content_cache[skill_name]
    
    root = skills_dir or _default_skills_dir()
    # Resolve by directory name first, then by frontmatter name
    cand = root / skill_name / "experience.md"
    if cand.exists():
        txt = cand.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(txt)
        result = {"found": True, "name": meta.get("name") or skill_name, "meta": meta, "content": body, "path": str(cand)}
        if use_cache:
            _skill_content_cache[skill_name] = result
        return result

    # Search by frontmatter name
    for s in list_skills(root):
        if s.name == skill_name:
            txt = s.path.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(txt)
            result = {"found": True, "name": s.name, "meta": meta, "content": body, "path": str(s.path)}
            if use_cache:
                _skill_content_cache[skill_name] = result
            return result

    return {"found": False, "name": skill_name, "meta": {}, "content": None, "path": None}


def smart_select_skills(
    code: str,
    query: str = "",
    *,
    max_skills: int = 2,
    max_content_chars: int = 4000,
    skills_dir: Path | None = None,
) -> dict[str, Any]:
    """Intelligently select and load skills based on code patterns and query.
    
    This is used for auto-injection: given the Python code the agent wants to run,
    automatically select the most relevant skills and return their content.
    
    Args:
        code: Python code to analyze for patterns
        query: User's question (optional, for additional context)
        max_skills: Maximum number of skills to include
        max_content_chars: Maximum total characters of skill content
        skills_dir: Override skills directory
    
    Returns:
        {
            "selected_skills": [skill_id, ...],
            "skill_summaries": [{name, description}, ...],
            "injected_content": "## Skill: name\n...",  # Combined skill content
            "total_chars": int,
        }
    """
    # Search with both code patterns and query
    skills = search_skills(query, k=max_skills * 2, code=code, skills_dir=skills_dir)
    
    if not skills:
        # Fallback: always include robust_df_checks for safety
        fallback = load_skill("robust_df_checks", skills_dir=skills_dir)
        if fallback.get("found"):
            content = fallback.get("content", "")[:max_content_chars]
            return {
                "selected_skills": ["robust_df_checks"],
                "skill_summaries": [{"name": "robust_df_checks", "description": fallback.get("meta", {}).get("description")}],
                "injected_content": f"## Skill: robust_df_checks\n{content}",
                "total_chars": len(content),
            }
        return {
            "selected_skills": [],
            "skill_summaries": [],
            "injected_content": "",
            "total_chars": 0,
        }
    
    selected_skills: list[str] = []
    skill_summaries: list[dict] = []
    content_parts: list[str] = []
    total_chars = 0
    
    for skill in skills[:max_skills]:
        skill_id = skill.path.parent.name
        loaded = load_skill(skill_id, skills_dir=skills_dir)
        
        if not loaded.get("found"):
            continue
        
        content = loaded.get("content", "")
        # Truncate if needed to stay within budget
        remaining = max_content_chars - total_chars
        if remaining <= 200:
            break
        
        if len(content) > remaining:
            content = content[:remaining] + "\n... (truncated)"
        
        selected_skills.append(skill_id)
        skill_summaries.append({
            "name": skill.name,
            "description": skill.description,
        })
        content_parts.append(f"## Skill: {skill.name}\n{content}")
        total_chars += len(content)
    
    return {
        "selected_skills": selected_skills,
        "skill_summaries": skill_summaries,
        "injected_content": "\n\n".join(content_parts),
        "total_chars": total_chars,
    }


def clear_skill_cache() -> None:
    """Clear the skill content cache."""
    global _skill_content_cache
    _skill_content_cache = {}


def get_skills_brief(skills_dir: Path | None = None) -> str:
    """Generate a compact category-grouped summary of skills for system prompt.
    
    Groups skills by domain to reduce token count while preserving discoverability.
    The agent should call `tool_search_and_load_skill(id)` for full content.
    """
    skills = list_skills(skills_dir)
    if not skills:
        return "No skills available."
    
    # Group skills by category using tags and naming conventions
    categories: dict[str, list[str]] = {}
    for skill in sorted(skills, key=lambda s: s.name):
        skill_id = skill.path.parent.name
        # Determine category from skill_id prefix or tags
        if skill_id.startswith("backtest_"):
            cat = "Backtest strategies"
        elif any(t in (skill.tags or []) for t in ["statistics", "regression", "forecast"]):
            cat = "Statistical analysis"
        elif skill_id in ("rolling_indicators", "momentum_breakout", "risk_metrics"):
            cat = "Technical indicators & risk"
        elif skill_id in ("adj_prices_and_returns", "merge_prices_and_valuation", "valuation_units"):
            cat = "Prices & valuation"
        elif skill_id in ("index_data", "index_returns_and_compare", "etf_data", "etf_nav_and_premium"):
            cat = "Index & ETF data"
        elif skill_id in ("finance_statements", "finance_statements_metrics"):
            cat = "Financial statements"
        elif skill_id in ("multi_stock_compare", "parallel_multi_stock"):
            cat = "Multi-stock analysis"
        else:
            cat = "Utilities"
        categories.setdefault(cat, []).append(skill_id)
    
    lines = []
    for cat, ids in sorted(categories.items()):
        lines.append(f"- **{cat}**: {', '.join(ids)}")
    
    return "\n".join(lines)


def select_top_skill_for_query(
    query: str,
    *,
    max_content_chars: int = 3000,
    skills_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Select the single most relevant skill for a user query.
    
    This is used for context injection: given the user's question,
    select the most relevant skill and return its content to be
    injected into the conversation context (after user message).
    
    Args:
        query: User's question
        max_content_chars: Maximum characters of skill content
        skills_dir: Override skills directory
    
    Returns:
        {
            "skill_id": str,
            "skill_name": str,
            "content": str,  # Full skill content (may be truncated)
            "score": int,
        }
        or None if no relevant skill found
    """
    if not query.strip():
        return None
    
    skills = list_skills(skills_dir)
    if not skills:
        return None
    
    # Score all skills by query
    scored = [(s, _score_skill(query, s, code="")) for s in skills]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Only return if score is above threshold (indicates relevance)
    top_skill, top_score = scored[0]
    if top_score < 3:  # Minimum relevance threshold (lowered for query-only selection)
        return None
    
    skill_id = top_skill.path.parent.name
    loaded = load_skill(skill_id, skills_dir=skills_dir)
    
    if not loaded.get("found"):
        return None
    
    content = loaded.get("content", "")
    if len(content) > max_content_chars:
        content = content[:max_content_chars] + "\n... (truncated)"
    
    return {
        "skill_id": skill_id,
        "skill_name": top_skill.name,
        "content": content,
        "score": top_score,
    }


