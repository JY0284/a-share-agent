"""Plugin registration — wires quant collectors and renderers into spiderman."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Called by spiderman's plugin loader at startup.

    Registers all quant collectors with the scheduler and
    quant renderers with the dispatcher.
    """
    from spiderman.scheduler import register_collector
    from spiderman.dispatcher import get_dispatcher

    from agent.notifications.collectors.portfolio_report import PortfolioReportCollector
    from agent.notifications.collectors.market_digest import MarketDigestCollector
    from agent.notifications.collectors.strategy_signal import StrategySignalCollector
    from agent.notifications.collectors.watchlist_alert import WatchlistAlertCollector
    from agent.notifications.renderers import QuantRenderer

    # Register collectors
    register_collector("portfolio_report", PortfolioReportCollector)
    register_collector("market_digest", MarketDigestCollector)
    register_collector("strategy_signal", StrategySignalCollector)
    register_collector("watchlist_alert", WatchlistAlertCollector)

    # Register renderers for all quant event types
    dispatcher = get_dispatcher()
    renderer = QuantRenderer()
    for event_type in (
        "portfolio_report",
        "market_digest",
        "strategy_signal",
        "watchlist_alert",
        "agent_result",
    ):
        dispatcher.register_renderer(event_type, renderer)

    logger.info("a-share-agent notification plugin registered")
