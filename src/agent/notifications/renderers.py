"""Quant-specific renderers — Jinja2 HTML rendering for A-share event types."""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from spiderman.core.events import Event
from spiderman.renderers.base import BaseRenderer, RenderedContent

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

# Human-readable subjects per event type
_SUBJECT_MAP = {
    "portfolio_report": "📊 投资组合日报 / Portfolio Report",
    "market_digest": "📈 市场速递 / Market Digest",
    "strategy_signal": "🎯 策略信号 / Strategy Signal",
    "watchlist_alert": "⚡ 自选股提醒 / Watchlist Alert",
    "agent_result": "🤖 分析报告 / Agent Analysis",
}


class QuantRenderer(BaseRenderer):
    """Renders quant events to HTML using Jinja2 templates."""

    def __init__(self) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
            autoescape=select_autoescape(["html"]),
        )

    def render(self, event: Event) -> RenderedContent:
        subject = _SUBJECT_MAP.get(event.event_type, f"[Quant] {event.event_type}")
        html = self._render_html(event)
        return RenderedContent(subject=subject, html=html)

    def _render_html(self, event: Event) -> str:
        template_name = f"{event.event_type}.html"
        try:
            template = self._env.get_template(template_name)
        except Exception:
            template = self._env.get_template("base.html")

        return template.render(
            event=event,
            payload=event.payload,
            event_type=event.event_type,
            user_id=event.user_id or "",
            created_at=event.created_at.strftime("%Y-%m-%d %H:%M"),
        )
