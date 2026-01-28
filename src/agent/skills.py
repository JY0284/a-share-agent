"""Skill system (agent-owned) for reusable analysis patterns.

Skills live at:
  a-share-agent/skills/<skill-name>/experience.md

Each experience.md contains:
- YAML frontmatter between --- markers (optional but recommended)
- Markdown body with guidance and examples
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Skill:
    name: str
    description: str | None
    tags: list[str]
    path: Path


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


def _score_skill(query: str, skill: Skill) -> int:
    """Very lightweight scorer that works ok for Chinese keywords + English."""
    q = (query or "").strip()
    if not q:
        return 0

    score = 0
    hay = " ".join(
        [
            skill.name,
            skill.description or "",
            " ".join(skill.tags),
            skill.path.parent.name,
        ]
    )

    # Strong boosts for direct substring matches (Chinese-friendly)
    for needle in [skill.name, skill.path.parent.name]:
        if needle and needle in q:
            score += 10

    # Token overlap (basic)
    q_tokens = [t for t in q.replace("，", " ").replace(",", " ").split() if t]
    for t in q_tokens:
        if t in hay:
            score += 3

    # Hand-tuned keyword boosts for common quant tasks
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
        "波动率": ["vol", "volatility", "波动率", "risk"],
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
    }
    for k, keys in boosts.items():
        if k in q:
            for kk in keys:
                if kk.lower() in hay.lower():
                    score += 2

    return score


def search_skills(query: str, *, k: int = 5, skills_dir: Path | None = None) -> list[Skill]:
    skills = list_skills(skills_dir)
    scored = [(s, _score_skill(query, s)) for s in skills]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, sc in scored[: max(0, int(k))] if sc > 0]


def load_skill(skill_name: str, *, skills_dir: Path | None = None) -> dict[str, Any]:
    root = skills_dir or _default_skills_dir()
    # Resolve by directory name first, then by frontmatter name
    cand = root / skill_name / "experience.md"
    if cand.exists():
        txt = cand.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(txt)
        return {"found": True, "name": meta.get("name") or skill_name, "meta": meta, "content": body, "path": str(cand)}

    # Search by frontmatter name
    for s in list_skills(root):
        if s.name == skill_name:
            txt = s.path.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(txt)
            return {"found": True, "name": s.name, "meta": meta, "content": body, "path": str(s.path)}

    return {"found": False, "name": skill_name, "meta": {}, "content": None, "path": None}

